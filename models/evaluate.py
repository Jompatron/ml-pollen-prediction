"""
Model evaluation: MAE, RMSE, quantile loss, calibration, per-horizon metrics.
All metrics computed in real units (grains/m³), not log space.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.feature_engineering import PollenFeatureEngineer

engineer = PollenFeatureEngineer()


def compute_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    mask = ~np.isnan(actuals) & ~np.isnan(predictions)
    return float(np.mean(np.abs(actuals[mask] - predictions[mask])))


def compute_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    mask = ~np.isnan(actuals) & ~np.isnan(predictions)
    return float(np.sqrt(np.mean((actuals[mask] - predictions[mask]) ** 2)))


def compute_quantile_loss(
    actuals: np.ndarray, quantile_preds: dict[float, np.ndarray]
) -> dict[float, float]:
    """Pinball loss per quantile. Lower is better."""
    losses = {}
    for q, preds in quantile_preds.items():
        mask = ~np.isnan(actuals) & ~np.isnan(preds)
        errors = actuals[mask] - preds[mask]
        losses[q] = float(np.mean(np.maximum(q * errors, (q - 1) * errors)))
    return losses


def compute_calibration(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """
    What fraction of actuals fall within the [lower, upper] interval?
    For a well-calibrated 80% interval (P10-P90), this should be ~0.80.
    """
    mask = ~np.isnan(actuals) & ~np.isnan(lower) & ~np.isnan(upper)
    coverage = np.mean((actuals[mask] >= lower[mask]) & (actuals[mask] <= upper[mask]))
    return float(coverage)


def risk_level(count: float, species: str) -> str:
    s = species.lower().replace("_pollen", "")
    if s in ("birch", "alder"):
        if count < 10: return "low"
        if count < 100: return "moderate"
        if count < 1000: return "high"
        return "very_high"
    if s == "grass":
        if count < 5: return "low"
        if count < 50: return "moderate"
        if count < 500: return "high"
        return "very_high"
    # mugwort
    if count < 5: return "low"
    if count < 30: return "moderate"
    if count < 100: return "high"
    return "very_high"


def evaluate_model(
    model,
    val_loader,
    species: str = "birch_pollen",
    save: bool = True,
) -> dict:
    """
    Run model on validation loader, compute all metrics.
    Returns metrics dict and optionally saves to artifacts/metrics/.
    """
    all_actuals = []
    all_p10 = []
    all_p50 = []
    all_p90 = []

    model.eval()
    for x, y in val_loader:
        with __import__("torch").no_grad():
            out = model(x)

        # out.prediction shape: (batch, prediction_length, n_quantiles)
        preds = out.prediction.cpu().numpy()
        targets = y[0].cpu().numpy()  # (batch, prediction_length)

        # Inverse transform from log1p space
        targets_real = engineer.inverse_transform_target(targets)
        p10_real = engineer.inverse_transform_target(preds[:, :, 0])  # Q10
        p50_real = engineer.inverse_transform_target(preds[:, :, 2])  # Q50
        p90_real = engineer.inverse_transform_target(preds[:, :, 4])  # Q90

        all_actuals.append(targets_real.flatten())
        all_p10.append(p10_real.flatten())
        all_p50.append(p50_real.flatten())
        all_p90.append(p90_real.flatten())

    actuals = np.concatenate(all_actuals)
    p10 = np.concatenate(all_p10)
    p50 = np.concatenate(all_p50)
    p90 = np.concatenate(all_p90)

    metrics = {
        "species": species,
        "evaluated_at": datetime.utcnow().isoformat(),
        "n_samples": int(len(actuals)),
        "mae": compute_mae(actuals, p50),
        "rmse": compute_rmse(actuals, p50),
        "calibration_80pct": compute_calibration(actuals, p10, p90),
        "quantile_losses": compute_quantile_loss(
            actuals,
            {0.1: p10, 0.5: p50, 0.9: p90},
        ),
    }

    logger.info(f"[{species}] MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | "
                f"Calibration={metrics['calibration_80pct']:.2%}")

    if save:
        out_path = settings.metrics_dir / f"{species}_eval_{datetime.utcnow().strftime('%Y%m%d')}.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Metrics saved to {out_path}")

    return metrics


def compare_to_previous(species: str) -> dict:
    """
    Load the two most recent evaluation files for a species and compute improvement.
    Returns {'previous_mae': ..., 'current_mae': ..., 'improvement_pct': ...}
    """
    files = sorted(settings.metrics_dir.glob(f"{species}_eval_*.json"))
    if len(files) < 2:
        logger.warning("Need at least 2 evaluation files to compare")
        return {}

    prev = json.loads(files[-2].read_text())
    curr = json.loads(files[-1].read_text())

    improvement = (prev["mae"] - curr["mae"]) / prev["mae"]
    result = {
        "previous_mae": prev["mae"],
        "current_mae": curr["mae"],
        "improvement_pct": improvement * 100,
        "better": improvement > 0,
    }
    logger.info(f"[{species}] MAE improvement: {improvement:.1%} "
                f"({prev['mae']:.2f} → {curr['mae']:.2f})")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-to-previous", action="store_true")
    args = parser.parse_args()

    if args.compare_to_previous:
        for species in settings.pollen_species:
            compare_to_previous(species)
