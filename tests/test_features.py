"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feature_engineering import PollenFeatureEngineer
from config.settings import settings


def make_sample_df(n_hours: int = 500) -> pd.DataFrame:
    """Create a minimal raw DataFrame for testing."""
    idx = pd.date_range("2024-04-01", periods=n_hours, freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)

    for species in settings.pollen_species:
        df[species] = np.random.exponential(scale=50, size=n_hours)

    # Some off-season nulls
    df.loc[df.index[:10], "birch_pollen"] = None

    df["temperature_2m"] = 10 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    df["relative_humidity_2m"] = 70 + 10 * np.random.randn(n_hours)
    df["wind_speed_10m"] = np.abs(5 + 2 * np.random.randn(n_hours))
    df["precipitation"] = np.random.exponential(0.1, size=n_hours)
    df["shortwave_radiation"] = np.maximum(0, 300 * np.sin(2 * np.pi * np.arange(n_hours) / 24))
    df["pm10"] = 20 + 5 * np.random.randn(n_hours)
    df["pm2_5"] = 10 + 3 * np.random.randn(n_hours)
    df["european_aqi"] = 25 + 5 * np.random.randn(n_hours)

    return df


def test_transform_runs_without_error():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    result = engineer.transform(raw)
    assert not result.empty


def test_pollen_nulls_filled_with_zero():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    raw["birch_pollen"] = None  # All null
    result = engineer.transform(raw)
    assert result["birch_pollen"].isna().sum() == 0
    assert (result["birch_pollen"] == 0).all()


def test_log1p_transform_non_negative():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    result = engineer.transform(raw)
    for species in settings.pollen_species:
        col = f"{species}_log1p"
        if col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negative values"


def test_cyclical_encodings_bounded():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    result = engineer.transform(raw)
    for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]:
        if col in result.columns:
            assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"


def test_no_lag_leakage():
    """Lag features must be shifted forward in time (past → present only)."""
    engineer = PollenFeatureEngineer()
    raw = make_sample_df(600)
    result = engineer.transform(raw)

    lag_col = "birch_pollen_lag_24h"
    if lag_col in result.columns:
        # At row i, lag_24h should equal the original birch_pollen 24h before.
        # After the encoder_length drop and time_idx reset, verify correlation is positive.
        correlation = result[lag_col].dropna().corr(result["birch_pollen"])
        # Should be reasonably correlated (pollen is autocorrelated)
        assert correlation > 0, "Lag feature has wrong sign — possible leakage"


def test_gdd_non_negative():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    result = engineer.transform(raw)
    if "gdd_cumulative" in result.columns:
        assert (result["gdd_cumulative"] >= 0).all()


def test_time_idx_contiguous():
    engineer = PollenFeatureEngineer()
    raw = make_sample_df()
    result = engineer.transform(raw)
    diffs = result["time_idx"].diff().dropna()
    assert (diffs == 1).all(), "time_idx is not contiguous"


def test_inverse_transform_roundtrip():
    engineer = PollenFeatureEngineer()
    values = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
    log_values = np.log1p(values)
    recovered = engineer.inverse_transform_target(log_values)
    np.testing.assert_allclose(recovered, values, rtol=1e-5)
