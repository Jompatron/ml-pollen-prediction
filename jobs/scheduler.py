"""
Local APScheduler runner — alternative to GitHub Actions for running jobs on a local machine or VPS.

Usage:
    python -m ml.jobs.scheduler
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from jobs.daily_collect import run as daily_run
from jobs.weekly_retrain import main as weekly_retrain


async def main():
    if not APSCHEDULER_AVAILABLE:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = AsyncIOScheduler()

    # Daily collection at 06:00 UTC (08:00 Stockholm)
    scheduler.add_job(
        daily_run,
        CronTrigger(hour=6, minute=0),
        id="daily_collect",
        name="Daily pollen data collection",
        replace_existing=True,
    )

    # Weekly retrain on Sunday at 02:00 UTC
    scheduler.add_job(
        lambda: asyncio.get_event_loop().run_in_executor(None, weekly_retrain),
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_retrain",
        name="Weekly model retraining",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started. Jobs: daily_collect (06:00 UTC), weekly_retrain (Sun 02:00 UTC)")

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    asyncio.run(main())
