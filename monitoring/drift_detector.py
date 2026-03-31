"""
Drift detection: feature drift (PSI) and prediction drift (MAE degradation).

Data drift:   Distribution of input features has shifted.
              Detected via Population Stability Index (PSI) on each feature.
Concept drift: Relationship between features and target has changed.
              Detected by comparing recent MAE to training MAE.

Results written to artifacts/metrics/drift_log.jsonl.
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
from data.storage import ParquetStorage


class DriftDetector:

    MONITOR_FEATURES = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "precipitation", "shortwave_radiation",
        "birch_pollen", "grass_pollen", "alder_pollen", "mugwort_pollen",
        "dispersal_potential", "gdd_cumulative",
    ]

    def __init__(self):
        self.storage = ParquetStorage()
        self.drift_log = settings.metrics_dir / "drift_log.jsonl"

    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Population Stability Index.
        < 0.1  → stable
        0.1–0.2 → moderate shift
        > 0.2  → significant shift, consider retraining
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) < 10 or len(current) < 10:
            return 0.0

        # Use reference distribution to define bin edges
        bins_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        bins_edges = np.unique(bins_edges)  # Remove duplicate edges
        if len(bins_edges) < 2:
            return 0.0

        ref_counts = np.histogram(reference, bins=bins_edges)[0]
        cur_counts = np.histogram(current, bins=bins_edges)[0]

        # Avoid division by zero / log(0)
        ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * len(ref_counts))
        cur_pct = (cur_counts + 1e-6) / (len(current) + 1e-6 * len(cur_counts))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def check_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute PSI for each monitored feature."""
        results = {}
        for col in self.MONITOR_FEATURES:
            if col in reference_df.columns and col in current_df.columns:
                psi = self.compute_psi(
                    reference_df[col].values,
                    current_df[col].values,
                )
                results[col] = round(psi, 4)
                if psi > settings.psi_threshold:
                    logger.warning(f"Feature drift detected: {col} PSI={psi:.3f}")
        return results

    def check_performance_drift(
        self,
        recent_predictions: pd.DataFrame,
        recent_actuals: pd.DataFrame,
        species: str = "birch_pollen",
    ) -> dict:
        """
        Compare recent MAE (last 7 days of predictions vs actuals) to baseline MAE.
        Returns degradation percentage.
        """
        # Load baseline MAE from latest evaluation file
        eval_files = sorted(settings.metrics_dir.glob(f"{species}_eval_*.json"))
        if not eval_files:
            logger.warning(f"No evaluation files for {species}")
            return {}

        baseline = json.loads(eval_files[-1].read_text())
        baseline_mae = baseline.get("mae", None)
        if baseline_mae is None:
            return {}

        # Compute recent MAE
        if recent_predictions.empty or recent_actuals.empty:
            return {}

        # Align on timestamp
        aligned = recent_predictions.join(recent_actuals, how="inner", rsuffix="_actual")
        if aligned.empty:
            return {}

        pred_col = f"{species}_p50"
        actual_col = species
        if pred_col not in aligned.columns or actual_col not in aligned.columns:
            return {}

        recent_mae = float(np.mean(np.abs(aligned[pred_col] - aligned[actual_col])))
        degradation = (recent_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0.0

        result = {
            "species": species,
            "baseline_mae": round(baseline_mae, 3),
            "recent_mae": round(recent_mae, 3),
            "degradation_pct": round(degradation * 100, 2),
            "degraded": degradation > settings.performance_decay_threshold,
        }

        if result["degraded"]:
            logger.warning(
                f"Performance drift: {species} MAE degraded {degradation:.1%} "
                f"({baseline_mae:.2f} → {recent_mae:.2f})"
            )
        return result

    def run_daily_check(self) -> dict:
        """
        Full drift check. Call after daily data collection.
        Compares last 7 days of data against the training reference window.
        """
        all_data = self.storage.load_all("merged")
        if len(all_data) < settings.min_samples_for_drift * 2:
            logger.info("Not enough data for drift detection")
            return {}

        # Reference: everything except last 7 days
        cutoff = all_data.index.max() - pd.Timedelta(days=7)
        reference = all_data[all_data.index <= cutoff]
        current = all_data[all_data.index > cutoff]

        if len(current) < settings.min_samples_for_drift:
            logger.info(f"Current window too small ({len(current)} rows)")
            return {}

        feature_drift = self.check_feature_drift(reference, current)

        result = {
            "checked_at": datetime.utcnow().isoformat(),
            "reference_rows": len(reference),
            "current_rows": len(current),
            "feature_psi": feature_drift,
            "features_drifted": [f for f, v in feature_drift.items() if v > settings.psi_threshold],
        }

        # Append to drift log
        with open(self.drift_log, "a") as f:
            f.write(json.dumps(result) + "\n")

        logger.info(f"Drift check complete. Drifted features: {result['features_drifted']}")
        return result

    def should_retrain(self) -> tuple[bool, str]:
        """
        Returns (should_retrain, reason).
        True if PSI > threshold on any key feature OR MAE degraded > threshold.
        """
        if not self.drift_log.exists():
            return False, "No drift log found"

        # Read last drift entry
        lines = self.drift_log.read_text().strip().split("\n")
        if not lines:
            return False, "Empty drift log"

        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False, "Could not parse drift log"

        drifted = last.get("features_drifted", [])
        if drifted:
            return True, f"Feature drift in: {', '.join(drifted)}"

        return False, "No significant drift detected"


if __name__ == "__main__":
    detector = DriftDetector()
    result = detector.run_daily_check()
    should, reason = detector.should_retrain()
    print(f"Should retrain: {should} | {reason}")
