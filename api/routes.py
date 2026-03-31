"""API route handlers."""

import json
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    ForecastResponse, HourlyForecast, CurrentConditions,
    HealthResponse, MetricsResponse, ModelMetrics,
    HistoryResponse, HistoryPoint,
)
from api.dependencies import get_forecaster, get_storage, get_tracker, get_drift_detector
from config.settings import settings
from models.predict import PollenForecaster

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    species: str | None = Query(None, description="Filter by species (e.g. birch_pollen)"),
    hours: int = Query(120, ge=1, le=120, description="Forecast horizon in hours"),
):
    """
    Probabilistic pollen forecast for Stockholm.
    Returns P10/P50/P90 quantiles per hour per species.
    """
    forecaster: PollenForecaster = get_forecaster()

    if not forecaster.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Run training first."
        )

    forecast_df = await forecaster.predict(hours=hours)

    if forecast_df.empty:
        raise HTTPException(status_code=500, detail="Forecast generation failed")

    # Filter by species if requested
    if species:
        species_key = species if "_pollen" in species else f"{species}_pollen"
        if species_key not in settings.pollen_species:
            raise HTTPException(status_code=400, detail=f"Unknown species: {species}")
        forecast_df = forecast_df[forecast_df["species"] == species_key]

    # Store for later accuracy evaluation
    tracker = get_tracker()
    tracker.store_prediction(forecast_df)

    items = []
    for _, row in forecast_df.iterrows():
        items.append(HourlyForecast(
            timestamp=row["timestamp"],
            species=row["species"],
            p10=max(0.0, row["p10"]),
            p25=max(0.0, row["p25"]),
            p50=max(0.0, row["p50"]),
            p75=max(0.0, row["p75"]),
            p90=max(0.0, row["p90"]),
            risk_level=row["risk_level"],
        ))

    model_info = forecaster.get_model_info()
    last_trained = None
    for info in model_info.values():
        if info["last_trained"]:
            t = datetime.fromisoformat(info["last_trained"])
            if last_trained is None or t > last_trained:
                last_trained = t

    return ForecastResponse(
        generated_at=datetime.utcnow(),
        location=settings.primary_location,
        forecast=items,
        model_version=last_trained.strftime("v%Y%m%d") if last_trained else "v0.0.0",
        source="ml_model",
    )


@router.get("/current", response_model=CurrentConditions)
async def get_current():
    """Current pollen levels from the latest stored observation."""
    storage = get_storage()
    all_data = storage.load_all("merged")

    if all_data.empty:
        raise HTTPException(status_code=503, detail="No data available")

    latest = all_data.iloc[-1]
    ts = all_data.index[-1]

    readings = {}
    for species in settings.pollen_species:
        if species in all_data.columns:
            readings[species] = max(0.0, float(latest.get(species, 0) or 0))

    dominant = max(readings, key=readings.get) if readings else "birch_pollen"

    from models.predict import _classify_risk
    max_count = max(readings.values()) if readings else 0
    overall = _classify_risk(max_count, dominant)

    return CurrentConditions(
        timestamp=ts.to_pydatetime(),
        location=settings.primary_location,
        species_readings=readings,
        dominant_species=dominant,
        overall_risk=overall,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Model version, last training date, data freshness, drift status."""
    forecaster: PollenForecaster = get_forecaster()
    storage = get_storage()
    drift_detector = get_drift_detector()

    model_info = forecaster.get_model_info()
    species_loaded = {s: info["n_members"] for s, info in model_info.items()}

    latest_ts = storage.get_latest_timestamp("merged")

    last_trained = None
    for info in model_info.values():
        if info["last_trained"]:
            t = datetime.fromisoformat(info["last_trained"])
            if last_trained is None or t > last_trained:
                last_trained = t

    should_retrain, reason = drift_detector.should_retrain()
    drift_status = f"retrain_needed: {reason}" if should_retrain else "stable"

    model_ready = forecaster.is_ready()
    status = "healthy" if model_ready else ("no_model" if not any(species_loaded.values()) else "degraded")

    return HealthResponse(
        status=status,
        model_ready=model_ready,
        species_loaded=species_loaded,
        last_data_update=latest_ts.to_pydatetime() if latest_ts else None,
        last_training_date=last_trained,
        drift_status=drift_status,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics():
    """Latest evaluation metrics from training + recent live performance."""
    tracker = get_tracker()
    recent = tracker.get_recent_metrics(n_days=30)

    # Load latest eval files per species
    latest_evals = []
    for species in settings.pollen_species:
        eval_files = sorted(settings.metrics_dir.glob(f"{species}_eval_*.json"))
        if eval_files:
            try:
                data = json.loads(eval_files[-1].read_text())
                latest_evals.append(ModelMetrics(
                    evaluated_at=datetime.fromisoformat(data["evaluated_at"]),
                    species=data["species"],
                    mae=data["mae"],
                    rmse=data["rmse"],
                    calibration_80pct=data.get("calibration_80pct", 0.0),
                    risk_level_accuracy=data.get("risk_level_accuracy"),
                ))
            except Exception as e:
                logger.warning(f"Could not parse eval file for {species}: {e}")

    return MetricsResponse(
        latest_metrics=latest_evals,
        recent_performance=recent,
    )


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    days: int = Query(7, ge=1, le=90, description="Number of days of history")
):
    """Actual observed pollen data for the last N days."""
    storage = get_storage()
    all_data = storage.load_all("merged")

    if all_data.empty:
        raise HTTPException(status_code=503, detail="No historical data available")

    cutoff = all_data.index.max() - pd.Timedelta(days=days)
    subset = all_data[all_data.index > cutoff]

    cols = settings.pollen_species + ["temperature_2m", "relative_humidity_2m"]
    subset = subset[[c for c in cols if c in subset.columns]]

    points = []
    for ts, row in subset.iterrows():
        points.append(HistoryPoint(
            timestamp=ts.to_pydatetime(),
            birch_pollen=float(row.get("birch_pollen", 0) or 0),
            grass_pollen=float(row.get("grass_pollen", 0) or 0),
            alder_pollen=float(row.get("alder_pollen", 0) or 0),
            mugwort_pollen=float(row.get("mugwort_pollen", 0) or 0),
            temperature_2m=float(row["temperature_2m"]) if "temperature_2m" in row and pd.notna(row["temperature_2m"]) else None,
            relative_humidity_2m=float(row["relative_humidity_2m"]) if "relative_humidity_2m" in row and pd.notna(row["relative_humidity_2m"]) else None,
        ))

    return HistoryResponse(
        location=settings.primary_location,
        days=days,
        data=points,
    )
