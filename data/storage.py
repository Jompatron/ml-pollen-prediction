"""Parquet-based local storage for time series data."""

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class ParquetStorage:
    """
    Stores time series data as monthly parquet partitions.
    Layout: artifacts/data/{category}/YYYY-MM.parquet
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.data_dir

    def _partition_path(self, category: str, year: int, month: int) -> Path:
        d = self.base_dir / category
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{year:04d}-{month:02d}.parquet"

    def save(self, df: pd.DataFrame, category: str) -> None:
        """Save a DataFrame, partitioned by month. Overwrites existing partitions."""
        if df.empty:
            logger.warning(f"Attempted to save empty DataFrame for {category}")
            return

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        for (year, month), group in df.groupby([df.index.year, df.index.month]):
            path = self._partition_path(category, int(year), int(month))
            group.to_parquet(path, compression="snappy")
            logger.debug(f"Saved {len(group)} rows → {path}")

    def load(self, category: str, start: date, end: date) -> pd.DataFrame:
        """Load data for a date range (inclusive)."""
        frames = []
        current = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)

        while current <= end_month:
            path = self._partition_path(category, current.year, current.month)
            if path.exists():
                frames.append(pd.read_parquet(path))
            else:
                logger.debug(f"No data for {category}/{current.year}-{current.month:02d}")
            # Advance to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        # Filter to exact date range
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return df.loc[start_ts:end_ts]

    def load_all(self, category: str) -> pd.DataFrame:
        """Load all available data for a category."""
        category_dir = self.base_dir / category
        if not category_dir.exists():
            return pd.DataFrame()

        paths = sorted(category_dir.glob("*.parquet"))
        if not paths:
            return pd.DataFrame()

        frames = [pd.read_parquet(p) for p in paths]
        return pd.concat(frames).sort_index()

    def get_latest_timestamp(self, category: str) -> pd.Timestamp | None:
        """Return the latest timestamp stored, or None if no data exists."""
        df = self.load_all(category)
        if df.empty:
            return None
        return df.index.max()

    def append_new(self, df: pd.DataFrame, category: str) -> int:
        """
        Append only rows newer than the latest stored timestamp.
        Returns the count of new rows written.
        """
        latest = self.get_latest_timestamp(category)
        if latest is not None:
            new_rows = df[df.index > latest]
        else:
            new_rows = df

        if new_rows.empty:
            logger.info(f"No new rows to append for {category}")
            return 0

        # Load existing month partitions that overlap, merge, re-save
        self.save(new_rows, category)
        logger.info(f"Appended {len(new_rows)} new rows to {category}")
        return len(new_rows)
