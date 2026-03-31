"""Tests for model training and baseline evaluation."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import PersistenceBaseline, SeasonalNaiveBaseline, LinearBaseline, evaluate_baselines
from data.feature_engineering import PollenFeatureEngineer
from config.settings import settings


def make_sample_features(n_hours: int = 600) -> pd.DataFrame:
    """Feature-engineered DataFrame ready for model input."""
    idx = pd.date_range("2024-03-01", periods=n_hours + settings.encoder_length, freq="h", tz="UTC")
    raw = pd.DataFrame(index=idx)
    for species in settings.pollen_species:
        raw[species] = np.random.exponential(scale=30, size=len(idx))
    raw["temperature_2m"] = 8 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24)
    raw["relative_humidity_2m"] = 70 + 5 * np.random.randn(len(idx))
    raw["wind_speed_10m"] = np.abs(5 + 2 * np.random.randn(len(idx)))
    raw["precipitation"] = np.random.exponential(0.05, size=len(idx))
    raw["shortwave_radiation"] = np.maximum(0, 200 * np.sin(2 * np.pi * np.arange(len(idx)) / 24))
    raw["pm10"] = 20 + 3 * np.random.randn(len(idx))
    raw["pm2_5"] = 10 + 2 * np.random.randn(len(idx))
    raw["european_aqi"] = 25 + 4 * np.random.randn(len(idx))

    engineer = PollenFeatureEngineer()
    return engineer.transform(raw)


def test_persistence_baseline_runs():
    df = make_sample_features()
    b = PersistenceBaseline()
    preds = b.predict(df, "birch_pollen")
    assert len(preds) == len(df)
    # First 24 rows should be NaN (shift=24)
    assert preds.iloc[:24].isna().all()
    assert preds.iloc[24:].notna().any()


def test_seasonal_naive_baseline_runs():
    df = make_sample_features()
    b = SeasonalNaiveBaseline()
    preds = b.predict(df, "birch_pollen")
    assert len(preds) == len(df)


def test_linear_baseline_fit_predict():
    df = make_sample_features()
    b = LinearBaseline()
    split = len(df) // 2
    b.fit(df.iloc[:split], "birch_pollen")
    preds = b.predict(df.iloc[split:])
    assert len(preds) == len(df) - split
    assert not preds.isna().any()
    assert (preds >= 0).all()  # Predictions should be non-negative (expm1 of log preds)


def test_evaluate_baselines_returns_all_metrics():
    df = make_sample_features()
    metrics = evaluate_baselines(df, "birch_pollen")
    assert "persistence_mae" in metrics
    assert "seasonal_naive_mae" in metrics
    assert all(v >= 0 for v in metrics.values())


def test_linear_baseline_non_negative_predictions():
    """All predictions should be >= 0 (no negative pollen counts)."""
    df = make_sample_features()
    b = LinearBaseline()
    b.fit(df.iloc[:400], "birch_pollen")
    preds = b.predict(df.iloc[400:])
    assert (preds >= 0).all()
