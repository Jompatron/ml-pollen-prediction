"""API endpoint tests."""

from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app
from api.dependencies import get_forecaster, get_storage, get_tracker, get_drift_detector


def make_mock_forecast_df():
    return pd.DataFrame([
        {
            "timestamp": datetime(2024, 4, 1, i),
            "species": "birch_pollen",
            "p10": 5.0, "p25": 10.0, "p50": 20.0, "p75": 35.0, "p90": 50.0,
            "risk_level": "moderate",
        }
        for i in range(5)
    ])


def make_mock_storage():
    mock = MagicMock()
    idx = pd.date_range("2024-04-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({
        "birch_pollen": [10.0] * 24,
        "grass_pollen": [0.0] * 24,
        "alder_pollen": [5.0] * 24,
        "mugwort_pollen": [0.0] * 24,
        "temperature_2m": [10.0] * 24,
        "relative_humidity_2m": [70.0] * 24,
    }, index=idx)
    mock.load_all.return_value = df
    mock.get_latest_timestamp.return_value = idx[-1]
    return mock


def make_mock_forecaster():
    mock = MagicMock()
    mock.is_ready.return_value = True
    mock.predict = AsyncMock(return_value=make_mock_forecast_df())
    mock.get_model_info.return_value = {
        "birch_pollen": {"n_members": 5, "last_trained": "2024-04-01T00:00:00"},
    }
    return mock


def make_mock_tracker():
    mock = MagicMock()
    mock.store_prediction = MagicMock()
    mock.get_recent_metrics.return_value = []
    return mock


def make_mock_drift():
    mock = MagicMock()
    mock.should_retrain.return_value = (False, "stable")
    return mock


@pytest.fixture
def client():
    app.dependency_overrides[get_forecaster] = make_mock_forecaster
    app.dependency_overrides[get_storage] = make_mock_storage
    app.dependency_overrides[get_tracker] = make_mock_tracker
    app.dependency_overrides[get_drift_detector] = make_mock_drift
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "endpoints" in resp.json()


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_ready" in data


def test_current_returns_200(client):
    resp = client.get("/api/v1/current")
    assert resp.status_code == 200
    data = resp.json()
    assert "species_readings" in data
    assert "overall_risk" in data


def test_history_returns_200(client):
    resp = client.get("/api/v1/history?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert isinstance(data["data"], list)


def test_forecast_returns_200(client):
    resp = client.get("/api/v1/forecast?hours=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 5
    first = data["forecast"][0]
    assert "p10" in first
    assert "p50" in first
    assert "p90" in first
    assert "risk_level" in first


def test_forecast_invalid_hours(client):
    resp = client.get("/api/v1/forecast?hours=999")
    assert resp.status_code == 422  # Validation error


def test_metrics_returns_200(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
