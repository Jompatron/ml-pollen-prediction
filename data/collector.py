"""Fetches pollen and weather data from Open-Meteo APIs."""

import asyncio
from datetime import date, timedelta

import httpx
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class OpenMeteoCollector:
    """Collects pollen and weather data from Open-Meteo APIs."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def fetch_pollen(
        self,
        lat: float,
        lng: float,
        start_date: date | None = None,
        end_date: date | None = None,
        past_days: int | None = None,
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch pollen + air quality data.
        Returns DataFrame with DatetimeIndex (UTC), one column per variable.
        """
        params = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(settings.pollen_species + settings.air_quality_features),
            "timezone": "UTC",
        }
        if start_date and end_date:
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
        else:
            params["past_days"] = past_days or settings.max_past_days
            params["forecast_days"] = forecast_days

        logger.debug(f"Fetching pollen data: {params}")
        resp = await self.client.get(settings.open_meteo_air_quality_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        return df

    async def fetch_weather(
        self,
        lat: float,
        lng: float,
        start_date: date | None = None,
        end_date: date | None = None,
        past_days: int | None = None,
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch weather features. Uses archive API for historical data
        (older than 92 days), forecast API for recent + future data.
        """
        cutoff = date.today() - timedelta(days=92)

        if start_date and end_date and end_date < cutoff:
            # Entirely historical — use archive
            return await self._fetch_weather_archive(lat, lng, start_date, end_date)
        elif start_date and end_date and start_date < cutoff:
            # Spans the boundary — fetch both and concatenate
            archive_df = await self._fetch_weather_archive(lat, lng, start_date, cutoff - timedelta(days=1))
            await asyncio.sleep(1)  # Rate limiting
            recent_df = await self._fetch_weather_recent(lat, lng, past_days=92, forecast_days=forecast_days)
            # Trim recent to requested range
            recent_df = recent_df[recent_df.index.date >= cutoff]
            return pd.concat([archive_df, recent_df]).sort_index()
        else:
            return await self._fetch_weather_recent(lat, lng, past_days=past_days, forecast_days=forecast_days)

    async def _fetch_weather_archive(
        self, lat: float, lng: float, start_date: date, end_date: date
    ) -> pd.DataFrame:
        # ERA5 archive does not provide forecast-only variables
        ARCHIVE_UNAVAILABLE = {"rain", "wind_gusts_10m"}
        archive_features = [f for f in settings.weather_features if f not in ARCHIVE_UNAVAILABLE]
        params = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(archive_features),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "UTC",
        }
        logger.debug(f"Fetching archive weather: {start_date} → {end_date}")
        resp = await self.client.get(settings.open_meteo_historical_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        return df

    async def _fetch_weather_recent(
        self,
        lat: float,
        lng: float,
        past_days: int | None = None,
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(settings.weather_features),
            "past_days": past_days or settings.max_past_days,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        logger.debug(f"Fetching recent weather")
        resp = await self.client.get(settings.open_meteo_weather_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        return df

    async def fetch_all(
        self,
        lat: float,
        lng: float,
        start_date: date | None = None,
        end_date: date | None = None,
        past_days: int | None = None,
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        """Fetch pollen + weather and merge on timestamp."""
        kwargs = dict(
            start_date=start_date,
            end_date=end_date,
            past_days=past_days,
            forecast_days=forecast_days,
        )
        pollen_df = await self.fetch_pollen(lat, lng, **kwargs)
        await asyncio.sleep(1)  # Rate limiting — be respectful to free tier
        weather_df = await self.fetch_weather(lat, lng, **kwargs)
        merged = pollen_df.join(weather_df, how="outer")
        return merged

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
