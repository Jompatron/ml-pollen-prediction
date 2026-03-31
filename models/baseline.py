"""
Simple baseline models for benchmarking.
TFT must beat these — if it doesn't, debug the features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class PersistenceBaseline:
    """
    Predicts that each future hour = the same-hour value 24 hours ago.
    Strong baseline during pollen season when day-to-day patterns repeat.
    """

    def predict(self, df: pd.DataFrame, species: str = "birch_pollen") -> pd.DataFrame:
        """
        Returns predictions aligned with the input index.
        prediction[t] = actual[t - 24h]
        """
        predictions = df[species].shift(24)
        return predictions.rename(f"{species}_persistence_pred")


class SeasonalNaiveBaseline:
    """
    Predicts using the value from 7 days ago (weekly seasonal pattern).
    Accounts for weekly cycles in human activity and reporting.
    """

    def predict(self, df: pd.DataFrame, species: str = "birch_pollen") -> pd.DataFrame:
        predictions = df[species].shift(168)  # 7 * 24 hours
        return predictions.rename(f"{species}_seasonal_naive_pred")


class LinearBaseline:
    """
    Ridge regression on a small set of hand-picked features.
    Fast sanity check: does linear regression beat persistence?
    """

    FEATURE_COLS = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "precipitation", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "dispersal_potential", "gdd_cumulative",
    ]

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.species = None

    def _get_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        # Add lag features
        lag_cols = [f"{self.species}_lag_24h", f"{self.species}_lag_168h"]
        available += [c for c in lag_cols if c in df.columns]
        return df[available].fillna(0).values, available

    def fit(self, df: pd.DataFrame, species: str = "birch_pollen"):
        self.species = species
        target = np.log1p(df[species].fillna(0).values)
        X, self.feature_names = self._get_features(df)
        X_scaled = self.scaler.fit_transform(X)
        # Remove rows where target is NaN
        mask = ~np.isnan(target)
        self.model.fit(X_scaled[mask], target[mask])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X, _ = self._get_features(df)
        X_scaled = self.scaler.transform(X)
        log_preds = self.model.predict(X_scaled)
        preds = np.expm1(log_preds)
        return pd.Series(preds, index=df.index, name=f"{self.species}_linear_pred")


def evaluate_baselines(
    df: pd.DataFrame, species: str = "birch_pollen"
) -> dict[str, float]:
    """
    Evaluate all baselines on the given DataFrame.
    Returns MAE for each baseline.
    """
    actuals = df[species].values
    results = {}

    # Persistence
    p = PersistenceBaseline()
    preds = p.predict(df, species).values
    mask = ~np.isnan(preds) & ~np.isnan(actuals)
    results["persistence_mae"] = float(np.mean(np.abs(preds[mask] - actuals[mask])))

    # Seasonal naive
    sn = SeasonalNaiveBaseline()
    preds = sn.predict(df, species).values
    mask = ~np.isnan(preds) & ~np.isnan(actuals)
    results["seasonal_naive_mae"] = float(np.mean(np.abs(preds[mask] - actuals[mask])))

    # Linear (train on first 70%, test on last 30%)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    if len(train_df) > 100 and len(test_df) > 10:
        lr = LinearBaseline()
        lr.fit(train_df, species)
        preds = lr.predict(test_df).values
        test_actuals = test_df[species].values
        mask = ~np.isnan(preds) & ~np.isnan(test_actuals)
        results["linear_mae"] = float(np.mean(np.abs(preds[mask] - test_actuals[mask])))

    return results
