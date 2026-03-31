"""Load trained ensemble models and generate probabilistic forecasts."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.collector import OpenMeteoCollector
from data.feature_engineering import PollenFeatureEngineer

try:
    from pytorch_forecasting import TemporalFusionTransformer
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


RISK_THRESHOLDS = {
    "birch_pollen":   {"low": 10,  "moderate": 100,  "high": 1000},
    "alder_pollen":   {"low": 10,  "moderate": 100,  "high": 1000},
    "grass_pollen":   {"low": 5,   "moderate": 50,   "high": 500},
    "mugwort_pollen": {"low": 5,   "moderate": 30,   "high": 100},
}


def _classify_risk(count: float, species: str) -> str:
    t = RISK_THRESHOLDS.get(species, RISK_THRESHOLDS["mugwort_pollen"])
    if count < t["low"]:
        return "low"
    if count < t["moderate"]:
        return "moderate"
    if count < t["high"]:
        return "high"
    return "very_high"


class PollenForecaster:
    """
    Loads trained ensemble models and generates forecasts.

    Prediction flow:
    1. Fetch latest 8 days of actual data + 5-day weather forecast
    2. Run feature engineering
    3. Pass through each ensemble member
    4. Aggregate: P10/P50/P90 across all members
    5. Inverse-transform to real pollen counts (grains/m³)
    6. Return structured forecast with uncertainty bands
    """

    def __init__(self, species_list: list[str] | None = None):
        self.species_list = species_list or settings.pollen_species
        self.engineer = PollenFeatureEngineer()
        self.models: dict[str, list] = {}  # species -> list of models
        self._load_models()

    def _load_models(self):
        if not PYTORCH_AVAILABLE:
            logger.warning("pytorch-forecasting not available — predictions will use fallback")
            return

        for species in self.species_list:
            species_dir = settings.models_dir / species
            if not species_dir.exists():
                logger.warning(f"No model directory for {species}")
                continue

            checkpoints = sorted(species_dir.glob("tft-*.ckpt"))
            if not checkpoints:
                logger.warning(f"No checkpoints found for {species}")
                continue

            # Load latest checkpoint per seed
            loaded = []
            seeds_seen = set()
            for ckpt in reversed(checkpoints):
                # Extract seed from filename like "tft-seed42-epoch003-val_loss0.1234.ckpt"
                parts = ckpt.stem.split("-")
                seed_part = next((p for p in parts if p.startswith("seed")), None)
                if seed_part and seed_part not in seeds_seen:
                    seeds_seen.add(seed_part)
                    try:
                        model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
                        model.eval()
                        loaded.append(model)
                        logger.info(f"Loaded {species} / {ckpt.name}")
                    except Exception as e:
                        logger.error(f"Failed to load {ckpt}: {e}")

            self.models[species] = loaded
            logger.info(f"{species}: {len(loaded)} ensemble members loaded")

    def is_ready(self) -> bool:
        return any(len(m) > 0 for m in self.models.values())

    def _predict_species(
        self, features_df: pd.DataFrame, species: str
    ) -> pd.DataFrame | None:
        """Run ensemble prediction for one species. Returns DataFrame with quantile columns."""
        ensemble = self.models.get(species, [])
        if not ensemble:
            return None

        target_col = f"{species}_log1p"
        if target_col not in features_df.columns:
            logger.error(f"Target column {target_col} not in features")
            return None

        from data.dataset import build_dataset, TIME_VARYING_KNOWN_REALS
        try:
            dataset = build_dataset(features_df, species=species, training=False)
        except Exception as e:
            logger.error(f"Failed to build dataset for {species}: {e}")
            return None

        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        all_member_preds = []
        with torch.no_grad():
            for model in ensemble:
                member_preds = []
                for x, _ in loader:
                    out = model(x)
                    member_preds.append(out.prediction.cpu().numpy())
                if member_preds:
                    all_member_preds.append(np.concatenate(member_preds, axis=0))

        if not all_member_preds:
            return None

        # Stack: (n_members, n_windows, prediction_length, n_quantiles)
        stacked = np.stack(all_member_preds, axis=0)

        # Flatten windows — take the last window (most recent forecast)
        last_window = stacked[:, -1, :, :]  # (n_members, prediction_length, n_quantiles)

        # Median across ensemble members
        median_preds = np.median(last_window, axis=0)  # (prediction_length, n_quantiles)

        # Inverse transform
        inv = self.engineer.inverse_transform_target

        forecast_hours = features_df.index[-1] + pd.Timedelta(hours=1)
        timestamps = pd.date_range(
            start=forecast_hours,
            periods=settings.prediction_length,
            freq="h",
            tz="UTC",
        )

        quantile_idx = {0.1: 0, 0.25: 1, 0.5: 2, 0.75: 3, 0.9: 4}
        rows = []
        for i, ts in enumerate(timestamps):
            row = {
                "timestamp": ts,
                "species": species,
                "p10": float(inv(median_preds[i, quantile_idx[0.1]])),
                "p25": float(inv(median_preds[i, quantile_idx[0.25]])),
                "p50": float(inv(median_preds[i, quantile_idx[0.5]])),
                "p75": float(inv(median_preds[i, quantile_idx[0.75]])),
                "p90": float(inv(median_preds[i, quantile_idx[0.9]])),
            }
            row["risk_level"] = _classify_risk(row["p50"], species)
            rows.append(row)

        return pd.DataFrame(rows).set_index("timestamp")

    async def predict(self, hours: int = 120) -> pd.DataFrame:
        """
        Full prediction pipeline. Returns DataFrame with columns:
        timestamp, species, p10, p25, p50, p75, p90, risk_level
        """
        # Need encoder_length (168h) + lag_drop (168h) + buffer = ~15 days of history
        past_days_needed = (settings.encoder_length * 2) // 24 + 2  # ~16 days
        async with OpenMeteoCollector() as collector:
            raw_data = await collector.fetch_all(
                settings.primary_lat,
                settings.primary_lng,
                past_days=past_days_needed,
                forecast_days=5,
            )

        features = self.engineer.transform(raw_data)

        all_forecasts = []
        for species in self.species_list:
            forecast = self._predict_species(features, species)
            if forecast is not None:
                all_forecasts.append(forecast.iloc[:hours])
            else:
                logger.warning(f"No prediction available for {species}")

        if not all_forecasts:
            return pd.DataFrame()

        return pd.concat(all_forecasts).reset_index()

    def get_model_info(self) -> dict:
        """Return metadata about loaded models."""
        info = {}
        for species in self.species_list:
            members = self.models.get(species, [])
            ckpts = sorted((settings.models_dir / species).glob("*.ckpt")) if (settings.models_dir / species).exists() else []
            latest_mtime = max((c.stat().st_mtime for c in ckpts), default=None)
            info[species] = {
                "n_members": len(members),
                "last_trained": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime else None,
            }
        return info
