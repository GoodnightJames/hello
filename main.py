"""
Trading Engine — Entry Point & Scheduler

Phase 3: Data pipeline + Signal engine + Execution.
- Loads configuration from YAML
- Initializes SQLite database and portfolio
- Schedules daily data ingestion via APScheduler
- Schedules signal scoring, decision engine, and paper execution
- Schedules end-of-day portfolio sync

Daily schedule (Mon-Fri, US/Eastern):
  08:00 — Data ingestion (daily OHLCV for universe)
  08:30 — Signal scoring + decision engine
  09:25 — Paper execution of approved decisions
  16:00 — End-of-day portfolio sync

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
from data.db import init_db, get_session
from data.ingestion import ingest_daily
from decision.engine import run_decision_engine
from execution.paper import execute_paper_decisions
from capital.manager import get_or_create_portfolio, save_portfolio_snapshot
from data.feature_store import get_price_history
from review.weekly_report import generate_weekly_report

load_dotenv()
logger = get_logger("main")

# Module-level state for passing decisions between scheduled jobs
_pending_decisions = []


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


def run_signal_and_decision():
    """Scheduled job: run signal scoring and decision engine."""
    global _pending_decisions

    if check_kill_switch():
        return

    logger.info("Running scheduled signal scoring + decision engine")
    try:
        decisions = run_decision_engine()
        _pending_decisions = decisions
        logger.info(
            "Scheduled decision engine complete",
            extra={
                "extra_data": {
                    "decision_count": len(decisions),
                    "summary": [
                        {"symbol": d["symbol"], "action": d["action"]}
                        for d in decisions
                    ],
                }
            },
        )
    except Exception as e:
        _pending_decisions = []
        logger.error(
            "Scheduled decision engine failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_paper_execution():
    """Scheduled job: execute pending decisions in paper mode."""
    global _pending_decisions

    if check_kill_switch():
        return

    if not _pending_decisions:
        logger.info("No pending decisions to execute")
        return

    logger.info(
        "Running paper execution",
        extra={"extra_data": {"decision_count": len(_pending_decisions)}},
    )
    try:
        results = execute_paper_decisions(_pending_decisions)
        _pending_decisions = []  # Clear after execution

        filled = [r for r in results if r.get("status") == "filled"]
        skipped = [r for r in results if r.get("status") == "skipped"]
        rejected = [r for r in results if r.get("status") == "rejected"]

        logger.info(
            "Paper execution complete",
            extra={
                "extra_data": {
                    "filled": len(filled),
                    "skipped": len(skipped),
                    "rejected": len(rejected),
                    "results": [
                        {"symbol": r["symbol"], "action": r["action"], "status": r["status"]}
                        for r in results
                    ],
                }
            },
        )
    except Exception as e:
        logger.error(
            "Paper execution failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_eod_sync():
    """Scheduled job: end-of-day portfolio sync and snapshot."""
    if check_kill_switch():
        return

    logger.info("Running end-of-day portfolio sync")
    try:
        session = get_session()
        portfolio = get_or_create_portfolio(session)
        positions = portfolio.get("positions", {})

        if positions:
            symbols = list(positions.keys())
            prices_df = get_price_history(symbols, lookback_days=2, session=session)
            if not prices_df.empty:
                latest = prices_df.iloc[-1]
                prices = {s: float(latest[s]) for s in latest.index if latest[s] > 0}
                save_portfolio_snapshot(session, portfolio["cash"], positions, prices)

        logger.info(
            "EOD sync complete",
            extra={
                "extra_data": {
                    "cash": portfolio["cash"],
                    "total_equity": portfolio["total_equity"],
                    "positions": len(positions),
                }
            },
        )
        session.close()
    except Exception as e:
        logger.error(
            "EOD sync failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_weekly_report():
    """Scheduled job: generate weekly performance report (Sunday)."""
    logger.info("Generating weekly report")
    try:
        report = generate_weekly_report()
        pnl = report.get("pnl", {})
        logger.info(
            "Weekly report generated",
            extra={
                "extra_data": {
                    "weekly_pnl": pnl.get("weekly_pnl_pct", 0),
                    "trades": len(report.get("trades", [])),
                    "risk_events": len(report.get("risk_events", [])),
                }
            },
        )
    except Exception as e:
        logger.error(
            "Weekly report failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def main():
    """Initialize engine and start scheduler."""
    logger.info("=" * 60)
    logger.info("Trading Engine starting (Data + Signals + Execution + Reporting)")
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

    # Initialize database and portfolio
    engine = init_db()
    logger.info("Database initialized", extra={"extra_data": {"url": str(engine.url)}})

    session = get_session()
    portfolio = get_or_create_portfolio(session)
    logger.info(
        "Portfolio state",
        extra={
            "extra_data": {
                "cash": portfolio["cash"],
                "total_equity": portfolio["total_equity"],
                "positions": len(portfolio.get("positions", {})),
            }
        },
    )
    session.close()

    # Parse schedule timing
    schedule = config["schedule"]
    tz = schedule["timezone"]

    def parse_time(key):
        h, m = schedule[key].split(":")
        return int(h), int(m)

    ing_h, ing_m = parse_time("data_ingestion")
    sig_h, sig_m = parse_time("signal_scoring")
    dec_h, dec_m = parse_time("decision_engine")
    eod_h, eod_m = parse_time("close_sync")

    # Set up scheduler
    scheduler = BlockingScheduler(timezone=tz)

    # Job 1: Daily data ingestion (08:00)
    scheduler.add_job(
        run_daily_ingestion,
        trigger=CronTrigger(day_of_week="mon-fri", hour=ing_h, minute=ing_m, timezone=tz),
        id="daily_ingestion",
        name="Daily OHLCV Data Ingestion",
    )

    # Job 2: Signal scoring + decision engine (08:30)
    scheduler.add_job(
        run_signal_and_decision,
        trigger=CronTrigger(day_of_week="mon-fri", hour=sig_h, minute=sig_m, timezone=tz),
        id="signal_and_decision",
        name="Signal Scoring + Decision Engine",
    )

    # Job 3: Paper execution (09:25 — just before market open)
    scheduler.add_job(
        run_paper_execution,
        trigger=CronTrigger(day_of_week="mon-fri", hour=dec_h, minute=dec_m, timezone=tz),
        id="paper_execution",
        name="Paper Order Execution",
    )

    # Job 4: End-of-day portfolio sync (16:00)
    scheduler.add_job(
        run_eod_sync,
        trigger=CronTrigger(day_of_week="mon-fri", hour=eod_h, minute=eod_m, timezone=tz),
        id="eod_sync",
        name="End-of-Day Portfolio Sync",
    )

    # Job 5: Weekly report (Sunday 10:00)
    scheduler.add_job(
        run_weekly_report,
        trigger=CronTrigger(day_of_week="sun", hour=10, minute=0, timezone=tz),
        id="weekly_report",
        name="Weekly Performance Report",
    )

    jobs = [
        {"id": "daily_ingestion", "trigger": f"Mon-Fri at {schedule['data_ingestion']} {tz}"},
        {"id": "signal_and_decision", "trigger": f"Mon-Fri at {schedule['signal_scoring']} {tz}"},
        {"id": "paper_execution", "trigger": f"Mon-Fri at {schedule['decision_engine']} {tz}"},
        {"id": "eod_sync", "trigger": f"Mon-Fri at {schedule['close_sync']} {tz}"},
        {"id": "weekly_report", "trigger": f"Sunday at 10:00 {tz}"},
    ]

    logger.info(
        "Scheduler configured",
        extra={"extra_data": {"jobs": jobs}},
    )

    logger.info("Starting scheduler — press Ctrl+C to exit")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
