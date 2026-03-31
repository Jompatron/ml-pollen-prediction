"""PyTorch Forecasting TimeSeriesDataSet builder."""

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False


# Columns known in the future (deterministic or from weather forecast)
TIME_VARYING_KNOWN_REALS = [
    "time_idx",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "birch_season", "grass_season", "alder_season", "mugwort_season",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "precipitation",
    "shortwave_radiation",
]

# Columns only known historically
TIME_VARYING_UNKNOWN_REALS_TEMPLATE = [
    "{species}_log1p",
    "{species}_lag_24h", "{species}_lag_48h", "{species}_lag_168h",
    "{species}_rolling_mean_24h", "{species}_rolling_max_24h",
    "{species}_rolling_std_24h",
    "pm10", "pm2_5", "european_aqi",
    "dispersal_potential", "rain_suppression",
    "gdd_cumulative", "radiation_daily_sum",
]


def _resolve_unknown_reals(species: str, df: pd.DataFrame) -> list[str]:
    """Fill in species-specific column names and filter to those that exist."""
    cols = []
    for template in TIME_VARYING_UNKNOWN_REALS_TEMPLATE:
        col = template.format(species=species)
        if col in df.columns:
            cols.append(col)
    return cols


def _resolve_known_reals(df: pd.DataFrame) -> list[str]:
    return [c for c in TIME_VARYING_KNOWN_REALS if c in df.columns]


def build_dataset(
    df: pd.DataFrame,
    species: str = "birch_pollen",
    training: bool = True,
    min_prediction_idx: int | None = None,
) -> "TimeSeriesDataSet":
    """
    Build a PyTorch Forecasting TimeSeriesDataSet from feature-engineered data.

    Args:
        df: Feature-engineered DataFrame with DatetimeIndex
        species: Target pollen species
        training: If True, allow randomization in sampling
        min_prediction_idx: For validation sets, the minimum time_idx to predict
    """
    if not PYTORCH_FORECASTING_AVAILABLE:
        raise ImportError(
            "pytorch-forecasting is not installed. "
            "Run: pip install pytorch-forecasting"
        )

    target_col = f"{species}_log1p"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Run feature engineering first.")

    unknown_reals = _resolve_unknown_reals(species, df)
    known_reals = _resolve_known_reals(df)

    # Ensure target is in unknown reals
    if target_col not in unknown_reals:
        unknown_reals = [target_col] + unknown_reals

    kwargs = dict(
        time_idx="time_idx",
        target=target_col,
        group_ids=["location"],
        max_encoder_length=settings.encoder_length,
        max_prediction_length=settings.prediction_length,
        static_categoricals=["location"],
        static_reals=["latitude", "longitude"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["location"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    if min_prediction_idx is not None:
        kwargs["min_prediction_idx"] = min_prediction_idx

    return TimeSeriesDataSet(df.reset_index(drop=True), **kwargs)


def build_val_dataset(
    training_dataset: "TimeSeriesDataSet",
    val_df: pd.DataFrame,
) -> "TimeSeriesDataSet":
    """Create a validation dataset from a training dataset (shares normalizer params)."""
    return TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df.reset_index(drop=True),
        stop_randomization=True,
    )
