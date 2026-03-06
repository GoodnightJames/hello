"""
Trading Engine — Entry Point & Scheduler

Phase 1: Data pipeline only.
- Loads configuration from YAML
- Initializes SQLite database
- Schedules daily data ingestion via APScheduler
- No trading logic — just data collection

Usage:
    python main.py
"""

import os
import sys

import yaml
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.logging import get_logger
from data.db import init_db
from data.ingestion import ingest_daily

load_dotenv()
logger = get_logger("main")


def load_config(config_path="config/settings.yaml"):
    """Load main configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_kill_switch():
    """Check if kill switch is activated."""
    kill = os.getenv("KILL_SWITCH", "false").lower()
    if kill == "true":
        logger.warning("KILL SWITCH ACTIVATED — halting all execution")
        return True
    return False


def run_daily_ingestion():
    """Scheduled job: run daily data ingestion."""
    if check_kill_switch():
        return

    logger.info("Running scheduled daily ingestion")
    try:
        count = ingest_daily()
        logger.info(
            "Scheduled ingestion complete",
            extra={"extra_data": {"new_bars": count}},
        )
    except Exception as e:
        logger.error(
            "Scheduled ingestion failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def main():
    """Initialize engine and start scheduler."""
    logger.info("=" * 60)
    logger.info("Trading Engine starting — Phase 1 (Data Pipeline)")
    logger.info("=" * 60)

    if check_kill_switch():
        logger.warning("Exiting due to kill switch")
        sys.exit(0)

    # Load config
    config = load_config()
    logger.info(
        "Config loaded",
        extra={
            "extra_data": {
                "active_strategy": config.get("active_strategy"),
                "universe_size": len(config["universe"]["equities"]),
                "timezone": config["schedule"]["timezone"],
            }
        },
    )

    # Initialize database
    engine = init_db()
    logger.info("Database initialized", extra={"extra_data": {"url": str(engine.url)}})

    # Parse schedule timing
    schedule = config["schedule"]
    ingestion_time = schedule["data_ingestion"]  # "08:00"
    hour, minute = ingestion_time.split(":")

    # Set up scheduler
    scheduler = BlockingScheduler(timezone=schedule["timezone"])

    # Daily data ingestion
    scheduler.add_job(
        run_daily_ingestion,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=int(hour),
            minute=int(minute),
            timezone=schedule["timezone"],
        ),
        id="daily_ingestion",
        name="Daily OHLCV Data Ingestion",
    )

    logger.info(
        "Scheduler configured",
        extra={
            "extra_data": {
                "jobs": [
                    {
                        "id": "daily_ingestion",
                        "trigger": f"Mon-Fri at {ingestion_time} {schedule['timezone']}",
                    }
                ]
            }
        },
    )

    logger.info("Starting scheduler — press Ctrl+C to exit")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
