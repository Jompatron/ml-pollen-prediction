"""
Model health check script.

Compares recent model predictions against actual observed pollen values
and reports rolling MAE, bias, interval coverage, and a retrain recommendation.

Usage:
    python scripts/check_model_health.py              # last 14 days (default)
    python scripts/check_model_health.py --days 30    # last 30 days
    python scripts/check_model_health.py --days 7     # last 7 days
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.storage import ParquetStorage
from data.feature_engineering import PollenFeatureEngineer
from models.predict import PollenForecaster, _classify_risk

# ── Thresholds for retrain recommendation ─────────────────────────────────────
# MAE is measured on log1p-transformed values (same scale the model uses)
MAE_RETRAIN_THRESHOLD = 0.50       # log1p units — ~65% relative error
BIAS_RETRAIN_THRESHOLD = 0.30      # systematic over/under-prediction
COVERAGE_RETRAIN_THRESHOLD = 0.60  # P10–P90 should cover ~80% of actuals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log1p(x: float) -> float:
    return float(np.log1p(max(x, 0)))


def _compute_metrics(actuals: pd.Series, p10: pd.Series, p50: pd.Series, p90: pd.Series) -> dict:
    """Compute MAE, RMSE, bias, and interval coverage on log1p scale."""
    log_actual = actuals.apply(_log1p)
    log_p50    = p50.apply(_log1p)
    log_p10    = p10.apply(_log1p)
    log_p90    = p90.apply(_log1p)

    errors = log_p50 - log_actual
    abs_errors = errors.abs()

    coverage_mask = (log_actual >= log_p10) & (log_actual <= log_p90)

    return {
        "n_hours":    int(len(actuals)),
        "mae":        float(abs_errors.mean()),
        "rmse":       float(np.sqrt((errors ** 2).mean())),
        "bias":       float(errors.mean()),          # positive = over-predicting
        "p90_coverage": float(coverage_mask.mean()), # fraction inside P10–P90 band
        "median_actual_grains": float(actuals.median()),
        "max_actual_grains":    float(actuals.max()),
    }


def _print_species_report(species: str, metrics: dict, flag: bool) -> None:
    status = "⚠  RETRAIN RECOMMENDED" if flag else "✓  OK"
    print(f"\n{'─'*55}")
    print(f"  {species.upper().replace('_', ' ')}   [{status}]")
    print(f"{'─'*55}")
    print(f"  Hours evaluated      : {metrics['n_hours']}")
    print(f"  MAE  (log1p scale)   : {metrics['mae']:.4f}   (threshold: {MAE_RETRAIN_THRESHOLD})")
    print(f"  RMSE (log1p scale)   : {metrics['rmse']:.4f}")
    print(f"  Bias (log1p scale)   : {metrics['bias']:+.4f}  {'(over-predicting)' if metrics['bias'] > 0.05 else '(under-predicting)' if metrics['bias'] < -0.05 else ''}")
    print(f"  P10–P90 coverage     : {metrics['p90_coverage']:.1%}  (threshold: >{COVERAGE_RETRAIN_THRESHOLD:.0%})")
    print(f"  Median actual        : {metrics['median_actual_grains']:.1f} grains/m³")
    print(f"  Max actual           : {metrics['max_actual_grains']:.1f} grains/m³")


def _flag_species(metrics: dict) -> bool:
    return (
        metrics["mae"]          > MAE_RETRAIN_THRESHOLD or
        abs(metrics["bias"])    > BIAS_RETRAIN_THRESHOLD or
        metrics["p90_coverage"] < COVERAGE_RETRAIN_THRESHOLD
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(days: int) -> None:
    end_date   = date.today() - timedelta(days=1)  # yesterday is the last full day
    start_date = end_date - timedelta(days=days - 1)

    print(f"\n{'='*55}")
    print(f"  POLLEN MODEL HEALTH CHECK")
    print(f"  Period: {start_date} → {end_date}  ({days} days)")
    print(f"{'='*55}")

    # ── Load stored actuals ────────────────────────────────────────────────────
    storage = ParquetStorage()
    raw_df = storage.load(category="merged", start=start_date, end=end_date)

    if raw_df.empty:
        print("\n[ERROR] No stored data found for this period.")
        print("  Run the daily collection job first:")
        print("  PYTHONPATH=. python -m jobs.daily_collect")
        sys.exit(1)

    engineer = PollenFeatureEngineer()
    features_df = engineer.transform(raw_df)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading ensemble models...")
    forecaster = PollenForecaster()

    if not forecaster.is_ready():
        print("\n[ERROR] No trained models found. Train models first.")
        sys.exit(1)

    model_info = forecaster.get_model_info()
    print(f"\nLoaded models:")
    for sp, info in model_info.items():
        members = info['n_members']
        trained = info['last_trained'] or 'unknown'
        print(f"  {sp:<20} {members} ensemble members  (last trained: {trained[:10]})")

    # ── Run rolling hindcast ───────────────────────────────────────────────────
    # Strategy: for each day in the window, we treat the preceding encoder_length
    # hours as context and predict the next prediction_length hours. We then
    # compare the first 24 hours of those predictions to the actual values.
    print(f"\nRunning hindcast over {days}-day window...")

    encoder_hours = settings.encoder_length   # 168 h
    pred_hours    = 24                        # compare first 24h of each forecast

    all_results: dict[str, list] = {sp: [] for sp in settings.pollen_species}

    # Slide a window one day at a time through the evaluation period
    eval_start = pd.Timestamp(start_date, tz="UTC") + pd.Timedelta(hours=encoder_hours)
    eval_end   = pd.Timestamp(end_date, tz="UTC")

    window_starts = pd.date_range(start=eval_start, end=eval_end, freq="24h")

    for ws in window_starts:
        context_start = ws - pd.Timedelta(hours=encoder_hours)
        context_end   = ws - pd.Timedelta(hours=1)
        actual_start  = ws
        actual_end    = ws + pd.Timedelta(hours=pred_hours - 1)

        context = features_df.loc[context_start:context_end]
        actuals_window = features_df.loc[actual_start:actual_end]

        if len(context) < encoder_hours or actuals_window.empty:
            continue

        for species in settings.pollen_species:
            target_col = f"{species}_log1p"
            actual_col = species

            if actual_col not in features_df.columns or target_col not in features_df.columns:
                continue

            forecast_df = forecaster._predict_species(context, species)
            if forecast_df is None or forecast_df.empty:
                continue

            forecast_window = forecast_df.iloc[:pred_hours]
            actual_values   = actuals_window[actual_col].reindex(forecast_window.index)

            valid = actual_values.notna() & forecast_window["p50"].notna()
            if valid.sum() < 6:
                continue

            for ts in forecast_window.index[valid]:
                all_results[species].append({
                    "timestamp": ts,
                    "actual":    actual_values[ts],
                    "p10":       forecast_window.loc[ts, "p10"],
                    "p50":       forecast_window.loc[ts, "p50"],
                    "p90":       forecast_window.loc[ts, "p90"],
                })

    # ── Compute and print metrics ──────────────────────────────────────────────
    any_flag = False
    flagged_species = []

    for species in settings.pollen_species:
        rows = all_results[species]
        if not rows:
            print(f"\n  {species}: not enough data to evaluate")
            continue

        df = pd.DataFrame(rows).set_index("timestamp")
        metrics = _compute_metrics(df["actual"], df["p10"], df["p50"], df["p90"])
        flag    = _flag_species(metrics)

        if flag:
            any_flag = True
            flagged_species.append(species)

        _print_species_report(species, metrics, flag)

    # ── Overall recommendation ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if any_flag:
        print(f"  RECOMMENDATION: RETRAIN")
        print(f"  Degraded species: {', '.join(flagged_species)}")
        print(f"\n  To retrain, run:")
        print(f"    PYTHONPATH=. python -m jobs.weekly_retrain")
        print(f"\n  Or trigger the GitHub Actions workflow manually:")
        print(f"    gh workflow run weekly-retrain.yml")
    else:
        print(f"  RECOMMENDATION: NO RETRAIN NEEDED")
        print(f"  All species within acceptable performance bounds.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pollen model health")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of past days to evaluate (default: 14)")
    args = parser.parse_args()

    if args.days < 8:
        print("Error: --days must be at least 8 (need 7 days of context + 1 day to evaluate)")
        sys.exit(1)

    asyncio.run(run(args.days))
