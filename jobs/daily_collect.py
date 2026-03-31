"""
Daily data collection job.
Fetches latest pollen + weather data and appends new rows to parquet storage.
Also runs drift detection.

Usage:
    python -m ml.jobs.daily_collect
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data.collector import OpenMeteoCollector
from data.storage import ParquetStorage
from monitoring.drift_detector import DriftDetector
from config.settings import settings


async def run():
    storage = ParquetStorage()
    collector = OpenMeteoCollector()

    logger.info("Starting daily data collection...")

    try:
        df = await collector.fetch_all(
            settings.primary_lat,
            settings.primary_lng,
            past_days=3,  # Fetch last 3 days to catch any gaps
            forecast_days=0,
        )

        new_rows = storage.append_new(df, "merged")
        logger.info(f"Daily collection complete: {new_rows} new rows")

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise
    finally:
        await collector.close()

    # Run drift detection
    logger.info("Running drift detection...")
    detector = DriftDetector()
    try:
        result = detector.run_daily_check()
        should_retrain, reason = detector.should_retrain()
        if should_retrain:
            logger.warning(f"Drift detected — retraining recommended: {reason}")
        else:
            logger.info("Drift check passed — no retraining needed")
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
