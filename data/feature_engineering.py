"""Feature engineering: transforms raw pollen + weather data into model-ready features."""

import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class PollenFeatureEngineer:
    """
    Transforms raw pollen + weather data into features for the TFT model.

    Feature categories:
    1. Target: pollen concentration per species (grains/m³)
    2. Time-varying KNOWN: temporal encodings, weather forecasts
    3. Time-varying UNKNOWN: actual pollen, lags, AQI
    4. Static covariates: location metadata
    """

    def transform(self, df: pd.DataFrame, location_name: str = "Stockholm") -> pd.DataFrame:
        """Full feature pipeline. Input: raw merged DataFrame. Output: model-ready DataFrame."""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # --- Handle nulls ---
        # Pollen is null outside season → fill with 0 (no pollen)
        for species in settings.pollen_species:
            if species in df.columns:
                df[species] = df[species].fillna(0).clip(lower=0)

        # Weather nulls → forward fill then backward fill
        weather_cols = [c for c in df.columns if c not in settings.pollen_species]
        df[weather_cols] = df[weather_cols].ffill().bfill()

        # --- Target transformation ---
        # Log1p: pollen counts are heavily right-skewed
        for species in settings.pollen_species:
            if species in df.columns:
                df[f"{species}_log1p"] = np.log1p(df[species])

        # --- Lag features (time-varying unknown) ---
        for species in settings.pollen_species:
            if species not in df.columns:
                continue
            for lag in [1, 3, 6, 12, 24, 48, 168]:
                df[f"{species}_lag_{lag}h"] = df[species].shift(lag)
            for window in [6, 24, 72]:
                df[f"{species}_rolling_mean_{window}h"] = (
                    df[species].rolling(window, min_periods=1).mean()
                )
                df[f"{species}_rolling_max_{window}h"] = (
                    df[species].rolling(window, min_periods=1).max()
                )
                df[f"{species}_rolling_std_{window}h"] = (
                    df[species].rolling(window, min_periods=1).std().fillna(0)
                )

        # --- Temporal features (time-varying known) ---
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_year"] = df.index.dayofyear
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Cyclical encoding — preserves circular nature of time
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # --- Season indicators ---
        df["birch_season"] = (
            (df["day_of_year"] >= 90) & (df["day_of_year"] <= 180)
        ).astype(int)
        df["grass_season"] = (
            (df["day_of_year"] >= 150) & (df["day_of_year"] <= 240)
        ).astype(int)
        df["alder_season"] = (
            (df["day_of_year"] >= 45) & (df["day_of_year"] <= 120)
        ).astype(int)
        df["mugwort_season"] = (
            (df["day_of_year"] >= 180) & (df["day_of_year"] <= 270)
        ).astype(int)

        # --- Weather interaction features ---
        if "temperature_2m" in df.columns and "wind_speed_10m" in df.columns:
            df["temp_x_wind"] = df["temperature_2m"] * df["wind_speed_10m"]

        if "relative_humidity_2m" in df.columns:
            df["dryness_index"] = (100 - df["relative_humidity_2m"]) / 100

        if all(c in df.columns for c in ["dryness_index", "wind_speed_10m", "temperature_2m"]):
            df["dispersal_potential"] = (
                df["dryness_index"]
                * df["wind_speed_10m"]
                * np.clip(df["temperature_2m"], 0, None)
            )

        if "precipitation" in df.columns:
            df["rain_suppression"] = np.where(df["precipitation"] > 0.5, 1, 0)

        # Growing Degree Days — proxy for phenological development
        # Birch releases pollen after accumulating enough thermal energy above 5°C
        if "temperature_2m" in df.columns:
            df["gdd_proxy"] = np.clip(df["temperature_2m"] - 5, 0, None)
            # Reset GDD accumulation at the start of each year
            df["year"] = df.index.year
            df["gdd_cumulative"] = df.groupby("year")["gdd_proxy"].cumsum()
            df.drop(columns=["year"], inplace=True)

        # Radiation daily sum
        if "shortwave_radiation" in df.columns:
            df["radiation_daily_sum"] = df["shortwave_radiation"].rolling(24, min_periods=1).sum()

        # --- Static covariates ---
        df["location"] = location_name
        df["latitude"] = settings.primary_lat
        df["longitude"] = settings.primary_lng

        # De-fragment before time_idx (many column inserts causes fragmentation)
        df = df.copy()

        # --- Integer time index for PyTorch Forecasting ---
        df["time_idx"] = np.arange(len(df))

        # --- Drop rows with NaN from lag features ---
        max_lag = 168
        if len(df) > max_lag:
            df = df.iloc[max_lag:].copy()
            # Reset time_idx to be contiguous after dropping
            df["time_idx"] = np.arange(len(df))

        logger.info(f"Feature engineering complete: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Convert log1p-transformed predictions back to grains/m³."""
        return np.expm1(np.maximum(values, 0))
