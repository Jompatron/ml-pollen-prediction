"""
Training pipeline with walk-forward cross-validation and ensemble training.

Walk-forward validation:
  Fold 1: Train months 1-6, validate month 7
  Fold 2: Train months 1-9, validate month 10
  Fold 3: Train months 1-12, validate month 13
  Final:  Train on all data

Ensemble: 5 models with different seeds — final prediction is their median.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.dataset import build_dataset, build_val_dataset
from data.feature_engineering import PollenFeatureEngineer
from data.storage import ParquetStorage
from models.tft_model import build_model
from models.evaluate import evaluate_model

try:
    # pytorch-forecasting 1.x uses the standalone `lightning` package,
    # not `pytorch_lightning`. Import from the same source to avoid type mismatches.
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        PYTORCH_FORECASTING_AVAILABLE = True
    except ImportError:
        PYTORCH_FORECASTING_AVAILABLE = False


def _make_loaders(train_dataset, val_dataset):
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=settings.batch_size, num_workers=0, shuffle=True
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=settings.batch_size * 2, num_workers=0
    )
    return train_loader, val_loader


def train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    species: str = "birch_pollen",
    seed: int = 42,
) -> tuple:
    """Train one TFT model. Returns (model, metrics, checkpoint_path)."""
    if not PYTORCH_FORECASTING_AVAILABLE:
        raise ImportError("pytorch-forecasting not installed")

    pl.seed_everything(seed, workers=True)

    target_col = f"{species}_log1p"
    assert target_col in train_df.columns, f"Missing {target_col} — run feature engineering first"

    train_dataset = build_dataset(train_df, species=species, training=True)
    val_dataset = build_val_dataset(train_dataset, val_df)

    train_loader, val_loader = _make_loaders(train_dataset, val_dataset)
    model = build_model(train_dataset)

    checkpoint_dir = settings.models_dir / species
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=settings.patience,
        mode="min",
        verbose=True,
    )
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"tft-seed{seed}-{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=settings.max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    metrics = evaluate_model(best_model, val_loader, species, save=True)
    metrics["checkpoint_path"] = str(best_path)
    metrics["seed"] = seed

    return best_model, metrics, best_path


def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    species: str = "birch_pollen",
) -> list:
    """Train ensemble of n_ensemble_members models. Returns list of models."""
    models = []
    all_metrics = []

    for i in range(settings.n_ensemble_members):
        seed = 42 + i * 17
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble member {i+1}/{settings.n_ensemble_members} | {species} | seed={seed}")
        logger.info(f"{'='*60}")

        model, metrics, ckpt = train_single_model(train_df, val_df, species=species, seed=seed)
        models.append(model)
        all_metrics.append(metrics)

    # Save ensemble summary
    summary = {
        "species": species,
        "trained_at": datetime.utcnow().isoformat(),
        "n_members": settings.n_ensemble_members,
        "members": all_metrics,
        "mean_mae": sum(m["mae"] for m in all_metrics) / len(all_metrics),
    }
    out = settings.metrics_dir / f"{species}_ensemble_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(f"Ensemble MAE: {summary['mean_mae']:.2f}")

    return models


def walk_forward_splits(df: pd.DataFrame, n_months_val: int = 3, n_folds: int = 3):
    """
    Yield (train_df, val_df) pairs for walk-forward validation.
    Each fold uses more history for training.
    """
    df = df.sort_index()
    months = sorted(set(zip(df.index.year, df.index.month)))

    if len(months) < n_months_val + 1:
        logger.warning("Not enough months for walk-forward validation, using single split")
        split_idx = int(len(df) * 0.8)
        yield df.iloc[:split_idx], df.iloc[split_idx:]
        return

    fold_size = max(1, (len(months) - n_months_val) // n_folds)

    for fold in range(n_folds):
        val_start_idx = fold_size * (fold + 1)
        if val_start_idx >= len(months):
            break

        train_months = months[:val_start_idx]
        val_months = months[val_start_idx:val_start_idx + n_months_val]

        def in_months(ts, month_list):
            return any(ts.year == y and ts.month == m for y, m in month_list)

        train_mask = [(ts.year, ts.month) in set(train_months) for ts in df.index]
        val_mask = [(ts.year, ts.month) in set(val_months) for ts in df.index]

        train_df = df[train_mask]
        val_df = df[val_mask]

        if len(train_df) > settings.encoder_length and len(val_df) > settings.prediction_length:
            logger.info(f"Fold {fold+1}: train={len(train_df)} rows, val={len(val_df)} rows")
            yield train_df, val_df


def train_all_species(full_df: pd.DataFrame):
    """Train ensembles for all pollen species."""
    engineer = PollenFeatureEngineer()

    # Single train/val split using last 20% as validation
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx:]

    for species in settings.pollen_species:
        logger.info(f"\n{'#'*60}\nTRAINING: {species}\n{'#'*60}")
        try:
            train_ensemble(train_df, val_df, species=species)
        except Exception as e:
            logger.error(f"Failed to train {species}: {e}")


def main():
    storage = ParquetStorage()
    raw_df = storage.load_all("merged")

    if raw_df.empty:
        logger.error("No data found. Run historical_backfill.py first.")
        return

    logger.info(f"Loaded {len(raw_df)} rows of raw data")

    engineer = PollenFeatureEngineer()
    features_df = engineer.transform(raw_df)
    logger.info(f"Feature engineering done: {features_df.shape}")

    train_all_species(features_df)
    logger.success("Training complete!")


if __name__ == "__main__":
    main()
