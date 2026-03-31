"""
Weekly retraining pipeline.
Loads all available data, runs feature engineering, trains ensemble, evaluates.

Usage:
    python -m ml.jobs.weekly_retrain
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data.storage import ParquetStorage
from data.feature_engineering import PollenFeatureEngineer
from models.train import train_all_species
from models.evaluate import compare_to_previous
from config.settings import settings


def main():
    logger.info("=== Weekly Retraining Pipeline ===")

    storage = ParquetStorage()
    raw_df = storage.load_all("merged")

    if raw_df.empty:
        logger.error("No data found. Run historical_backfill.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(raw_df)} rows spanning {raw_df.index.min()} → {raw_df.index.max()}")

    engineer = PollenFeatureEngineer()
    features_df = engineer.transform(raw_df)
    logger.info(f"Feature engineering done: {features_df.shape}")

    train_all_species(features_df)

    # Compare to previous models
    logger.info("=== Comparing to previous models ===")
    for species in settings.pollen_species:
        try:
            compare_to_previous(species)
        except Exception as e:
            logger.warning(f"Could not compare {species}: {e}")

    logger.success("Weekly retraining complete!")


if __name__ == "__main__":
    main()
