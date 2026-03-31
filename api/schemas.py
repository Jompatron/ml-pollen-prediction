"""Pydantic request/response models for the FastAPI."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HourlyForecast(BaseModel):
    timestamp: datetime
    species: str
    p10: float = Field(ge=0, description="10th percentile (grains/m³)")
    p25: float = Field(ge=0)
    p50: float = Field(ge=0, description="Median forecast")
    p75: float = Field(ge=0)
    p90: float = Field(ge=0, description="90th percentile")
    risk_level: Literal["low", "moderate", "high", "very_high"]


class ForecastResponse(BaseModel):
    generated_at: datetime
    location: str
    forecast: list[HourlyForecast]
    model_version: str
    source: Literal["ml_model", "cams_baseline"] = "ml_model"


class CurrentConditions(BaseModel):
    timestamp: datetime
    location: str
    species_readings: dict[str, float]  # species -> current grains/m³
    dominant_species: str
    overall_risk: Literal["low", "moderate", "high", "very_high"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "no_model"]
    model_ready: bool
    species_loaded: dict[str, int]  # species -> n ensemble members
    last_data_update: datetime | None
    last_training_date: datetime | None
    drift_status: str


class ModelMetrics(BaseModel):
    evaluated_at: datetime
    species: str
    mae: float
    rmse: float
    calibration_80pct: float
    risk_level_accuracy: float | None = None


class MetricsResponse(BaseModel):
    latest_metrics: list[ModelMetrics]
    recent_performance: list[dict]


class HistoryPoint(BaseModel):
    timestamp: datetime
    birch_pollen: float | None
    grass_pollen: float | None
    alder_pollen: float | None
    mugwort_pollen: float | None
    temperature_2m: float | None
    relative_humidity_2m: float | None


class HistoryResponse(BaseModel):
    location: str
    days: int
    data: list[HistoryPoint]
