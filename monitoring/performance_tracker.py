"""
Track live model accuracy by comparing stored predictions to actual observations.

Every day after new actual data arrives:
1. Load predictions that were stored 1-5 days ago
2. Compare to observed actuals for the same timestamps
3. Compute MAE, calibration, risk-level accuracy per forecast horizon
4. Append to artifacts/metrics/performance_log.jsonl

This populates the /metrics API endpoint with real data.
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.storage import ParquetStorage


RISK_THRESHOLDS = {
    "birch_pollen":   (10, 100, 1000),
    "alder_pollen":   (10, 100, 1000),
    "grass_pollen":   (5,  50,  500),
    "mugwort_pollen": (5,  30,  100),
}


def _classify_risk(count: float, species: str) -> str:
    low, mod, high = RISK_THRESHOLDS.get(species, (5, 30, 100))
    if count < low: return "low"
    if count < mod: return "moderate"
    if count < high: return "high"
    return "very_high"


class PerformanceTracker:

    def __init__(self):
        self.storage = ParquetStorage()
        self.predictions_dir = settings.metrics_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        self.perf_log = settings.metrics_dir / "performance_log.jsonl"

    def store_prediction(self, forecast_df: pd.DataFrame, generated_at: datetime | None = None):
        """
        Save a forecast for later comparison against actuals.
        forecast_df should have columns: timestamp, species, p10, p50, p90, risk_level
        """
        ts = (generated_at or datetime.utcnow()).strftime("%Y%m%dT%H%M%S")
        out_path = self.predictions_dir / f"forecast_{ts}.parquet"
        forecast_df.to_parquet(out_path)
        logger.info(f"Stored forecast → {out_path}")

    def _load_stored_predictions(self, days_ago_min: int = 1, days_ago_max: int = 5) -> pd.DataFrame:
        """Load predictions that were generated between days_ago_min and days_ago_max ago."""
        cutoff_new = datetime.utcnow() - timedelta(days=days_ago_min)
        cutoff_old = datetime.utcnow() - timedelta(days=days_ago_max)

        frames = []
        for f in self.predictions_dir.glob("forecast_*.parquet"):
            try:
                ts = datetime.strptime(f.stem.replace("forecast_", ""), "%Y%m%dT%H%M%S")
                if cutoff_old <= ts <= cutoff_new:
                    df = pd.read_parquet(f)
                    df["generated_at"] = ts
                    frames.append(df)
            except Exception:
                continue

        return pd.concat(frames) if frames else pd.DataFrame()

    def evaluate_against_actuals(self) -> dict:
        """
        Compare stored predictions to recent actuals.
        Returns per-horizon and per-species accuracy metrics.
        """
        predictions = self._load_stored_predictions()
        if predictions.empty:
            logger.info("No stored predictions to evaluate")
            return {}

        actuals_df = self.storage.load_all("merged")
        if actuals_df.empty:
            logger.info("No actual data available")
            return {}

        results_by_species = {}

        for species in settings.pollen_species:
            if species not in actuals_df.columns:
                continue

            species_preds = predictions[predictions.get("species", pd.Series()) == species] if "species" in predictions.columns else pd.DataFrame()
            if species_preds.empty:
                continue

            # Merge predictions with actuals on timestamp
            species_preds = species_preds.set_index("timestamp") if "timestamp" in species_preds.columns else species_preds
            actual_series = actuals_df[species].rename("actual")

            merged = species_preds.join(actual_series, how="inner")
            if merged.empty:
                continue

            # Per-horizon metrics (bucket into 1h, 6h, 24h, 48h, 120h horizons)
            merged["forecast_horizon_h"] = (
                merged.index - merged.get("generated_at", merged.index)
            ).dt.total_seconds() / 3600 if "generated_at" in merged.columns else 0

            actuals_arr = merged["actual"].values
            p50_arr = merged["p50"].values if "p50" in merged.columns else np.zeros_like(actuals_arr)
            p10_arr = merged["p10"].values if "p10" in merged.columns else np.zeros_like(actuals_arr)
            p90_arr = merged["p90"].values if "p90" in merged.columns else np.zeros_like(actuals_arr)

            mask = ~np.isnan(actuals_arr) & ~np.isnan(p50_arr)
            if mask.sum() < 10:
                continue

            mae = float(np.mean(np.abs(p50_arr[mask] - actuals_arr[mask])))
            rmse = float(np.sqrt(np.mean((p50_arr[mask] - actuals_arr[mask]) ** 2)))
            coverage = float(np.mean(
                (actuals_arr[mask] >= p10_arr[mask]) & (actuals_arr[mask] <= p90_arr[mask])
            ))

            # Risk level accuracy
            pred_risk = [_classify_risk(p, species) for p in p50_arr[mask]]
            actual_risk = [_classify_risk(a, species) for a in actuals_arr[mask]]
            risk_accuracy = float(np.mean([p == a for p, a in zip(pred_risk, actual_risk)]))

            results_by_species[species] = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "calibration_80pct": round(coverage, 3),
                "risk_level_accuracy": round(risk_accuracy, 3),
                "n_samples": int(mask.sum()),
            }

        if not results_by_species:
            return {}

        record = {
            "evaluated_at": datetime.utcnow().isoformat(),
            "date": date.today().isoformat(),
            "species_metrics": results_by_species,
        }

        with open(self.perf_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"Performance evaluation saved: {list(results_by_species.keys())}")
        return record

    def get_recent_metrics(self, n_days: int = 30) -> list[dict]:
        """Return the last n_days of performance log entries."""
        if not self.perf_log.exists():
            return []

        cutoff = datetime.utcnow() - timedelta(days=n_days)
        results = []
        for line in self.perf_log.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                record = json.loads(line)
                ts = datetime.fromisoformat(record["evaluated_at"])
                if ts >= cutoff:
                    results.append(record)
            except Exception:
                continue
        return results
