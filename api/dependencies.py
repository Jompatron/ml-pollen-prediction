"""Shared dependencies: model loading, caching."""

from functools import lru_cache
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predict import PollenForecaster
from data.storage import ParquetStorage
from monitoring.performance_tracker import PerformanceTracker
from monitoring.drift_detector import DriftDetector


@lru_cache(maxsize=1)
def get_forecaster() -> PollenForecaster:
    logger.info("Loading PollenForecaster...")
    return PollenForecaster()


@lru_cache(maxsize=1)
def get_storage() -> ParquetStorage:
    return ParquetStorage()


@lru_cache(maxsize=1)
def get_tracker() -> PerformanceTracker:
    return PerformanceTracker()


@lru_cache(maxsize=1)
def get_drift_detector() -> DriftDetector:
    return DriftDetector()
