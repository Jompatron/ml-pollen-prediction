"""
One-time script to backfill historical pollen + weather data.

Strategy:
- Open-Meteo CAMS pollen data available from ~Oct 2023 onward
- Fetch in 90-day chunks with 1s rate-limit delay
- ERA5 weather archive goes back to 1940 — fetch same period
- Save as monthly parquet partitions in artifacts/data/

Usage:
    python -m ml.data.historical_backfill
    python -m ml.data.historical_backfill --start 2023-10-01 --end 2024-12-31
"""

import asyncio
import argparse
from datetime import date, timedelta

from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collector import OpenMeteoCollector
from data.storage import ParquetStorage
from config.settings import settings

CHUNK_DAYS = 89  # Stay well under 90-day API limit
POLLEN_DATA_START = date(2023, 10, 1)  # Earliest CAMS pollen data


def date_chunks(start: date, end: date, chunk_days: int):
    """Yield (chunk_start, chunk_end) pairs."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


async def backfill(start: date, end: date):
    storage = ParquetStorage()

    # Check what we already have
    latest = storage.get_latest_timestamp("merged")
    if latest is not None:
        already_through = latest.date()
        if already_through >= end:
            logger.info(f"Already have data through {already_through}. Nothing to do.")
            return
        # Resume from where we left off
        resume_from = already_through + timedelta(days=1)
        logger.info(f"Resuming backfill from {resume_from}")
        start = max(start, resume_from)

    chunks = list(date_chunks(start, end, CHUNK_DAYS))
    logger.info(f"Backfilling {start} → {end} in {len(chunks)} chunks")

    async with OpenMeteoCollector() as collector:
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Chunk {i+1}/{len(chunks)}: {chunk_start} → {chunk_end}")
            try:
                df = await collector.fetch_all(
                    settings.primary_lat,
                    settings.primary_lng,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    forecast_days=0,
                )
                if not df.empty:
                    storage.save(df, "merged")
                    logger.info(f"  Saved {len(df)} rows")
                else:
                    logger.warning(f"  Empty response for chunk {chunk_start} → {chunk_end}")
            except Exception as e:
                logger.error(f"  Failed chunk {chunk_start} → {chunk_end}: {e}")

            if i < len(chunks) - 1:
                await asyncio.sleep(5)  # Rate limiting — archive API needs breathing room

    logger.success(f"Backfill complete. Total stored: {storage.load_all('merged').shape}")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical pollen data")
    parser.add_argument("--start", default=POLLEN_DATA_START.isoformat(),
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    asyncio.run(backfill(start, end))


if __name__ == "__main__":
    main()
