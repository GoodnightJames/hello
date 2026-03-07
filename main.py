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
from apscheduler.triggers.interval import IntervalTrigger

from core.logging import get_logger
from core.market_calendar import is_market_open
from data.db import init_db, get_session
from data.ingestion import ingest_daily, ingest_crypto
from decision.engine import run_decision_engine, run_daily_decision_engine
from execution.paper import execute_paper_decisions
from capital.manager import get_or_create_portfolio, save_portfolio_snapshot, record_deposit
from data.feature_store import get_price_history
from review.weekly_report import generate_weekly_report
from execution.paper import sync_portfolio_from_alpaca

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


def run_weekly_deposit():
    """Scheduled job: record weekly $100 deposit (Monday 07:30 ET)."""
    if check_kill_switch():
        return

    if not is_market_open():
        logger.info("Market closed today (holiday) — skipping deposit")
        return

    logger.info("Processing weekly deposit")
    try:
        session = get_session()

        # Get current performance mode so deposit respects conservative buffering
        from risk.enforcer import load_risk_params
        from performance.manager import select_risk_mode, get_mode_params
        risk_params = load_risk_params()
        mode_result = select_risk_mode(session, risk_params)
        mode_params = get_mode_params(risk_params, mode_result["mode"])

        portfolio = record_deposit(session, mode_params=mode_params)
        session.commit()

        logger.info(
            "Weekly deposit complete",
            extra={
                "extra_data": {
                    "cash": portfolio["cash"],
                    "total_equity": portfolio["total_equity"],
                    "risk_mode": mode_result["mode"],
                }
            },
        )
        session.close()
    except Exception as e:
        logger.error(
            "Weekly deposit failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_performance_check():
    """Scheduled job: log current performance mode (daily 08:15 ET)."""
    if check_kill_switch():
        return

    if not is_market_open():
        return

    try:
        session = get_session()
        from risk.enforcer import load_risk_params
        from performance.manager import select_risk_mode

        risk_params = load_risk_params()
        mode_result = select_risk_mode(session, risk_params)

        logger.info(
            "Performance mode check",
            extra={
                "extra_data": {
                    "mode": mode_result["mode"],
                    "reason": mode_result.get("reason", ""),
                    "metrics": mode_result.get("metrics", {}),
                }
            },
        )
        session.close()
    except Exception as e:
        logger.error(
            "Performance check failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_daily_ingestion():
    """Scheduled job: run daily data ingestion."""
    if check_kill_switch():
        return

    if not is_market_open():
        logger.info("Market closed today (holiday) — skipping ingestion")
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
    """
    Scheduled job: run signal scoring and decision engine.

    Mon-Thu: daily scan — exit losers with 3m breakdown, fill replacements
    Friday:  weekly rebalance — full re-rank with new position sizing
    """
    global _pending_decisions

    if check_kill_switch():
        return

    if not is_market_open():
        logger.info("Market closed today (holiday) — skipping signals")
        return

    from datetime import datetime as dt
    weekday = dt.now().weekday()  # 0=Mon, 4=Fri
    is_rebalance_day = weekday == 4

    if is_rebalance_day:
        logger.info("Running WEEKLY rebalance (Friday)")
        engine_fn = run_decision_engine
    else:
        logger.info("Running DAILY scan (Mon-Thu)")
        engine_fn = run_daily_decision_engine

    try:
        decisions = engine_fn()
        _pending_decisions = decisions
        logger.info(
            "Scheduled decision engine complete",
            extra={
                "extra_data": {
                    "mode": "weekly_rebalance" if is_rebalance_day else "daily_scan",
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

    if not is_market_open():
        logger.info("Market closed today (holiday) — skipping execution")
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
    """Scheduled job: end-of-day portfolio sync from Alpaca."""
    if check_kill_switch():
        return

    if not is_market_open():
        logger.info("Market closed today (holiday) — skipping EOD sync")
        return

    logger.info("Running end-of-day portfolio sync (Alpaca)")
    try:
        session = get_session()
        portfolio = sync_portfolio_from_alpaca(session)
        session.commit()

        logger.info(
            "EOD sync complete (Alpaca)",
            extra={
                "extra_data": {
                    "cash": portfolio["cash"],
                    "total_equity": portfolio["total_equity"],
                    "positions": len(portfolio.get("positions", {})),
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


def run_crypto_ingestion():
    """Scheduled job: run crypto data ingestion (24/7, no market calendar)."""
    if check_kill_switch():
        return

    logger.info("Running scheduled crypto ingestion")
    try:
        count = ingest_crypto()
        logger.info(
            "Scheduled crypto ingestion complete",
            extra={"extra_data": {"new_bars": count}},
        )
    except Exception as e:
        logger.error(
            "Scheduled crypto ingestion failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )


def run_crypto_dca_cycle():
    """
    Scheduled job: crypto DCA — buy fixed dollar amounts on schedule.

    Runs 24/7, every N hours. No momentum check, no regime filter.
    Just dollar-cost average into crypto using weighted allocation.
    """
    if check_kill_switch():
        return

    logger.info("=" * 40)
    logger.info("Crypto DCA cycle starting")
    logger.info("=" * 40)

    try:
        from strategies.crypto_dca_v1 import CryptoDCAStrategy
        from data.db import get_session as get_db_session
        from capital.manager import get_sleeve_deployable_cash, SLEEVE_CRYPTO

        strategy = CryptoDCAStrategy()

        # Check if crypto sleeve has enough capital for DCA
        session = get_db_session()
        sleeve_cash = get_sleeve_deployable_cash(session, SLEEVE_CRYPTO)
        session.close()

        total_deploy = strategy.dollars_per_cycle

        if sleeve_cash < total_deploy:
            logger.info(
                f"Crypto DCA: insufficient sleeve cash (${sleeve_cash:.2f} < ${total_deploy:.2f})"
            )
            return

        # Generate DCA buy signals (always buys, no filters)
        signals = strategy.generate_signals()

        if not signals:
            logger.info("Crypto DCA: no signals generated")
            return

        # Convert signals directly to decisions (no regime filter for DCA)
        decisions = []
        for sig in signals:
            decisions.append({
                "symbol": sig["symbol"],
                "action": "BUY",
                "reason": sig["metadata"]["reason"],
                "signal_type": "BUY",
                "signal_id": None,
                "position_multiplier": 1.0,
                "signal_strength": 1.0,
                "risk_approved": True,
                "dca_dollar_amount": sig["metadata"]["dollar_amount"],
            })

        logger.info(
            "Crypto DCA decisions ready",
            extra={
                "extra_data": {
                    "decisions": len(decisions),
                    "total_deploy": total_deploy,
                    "summary": [
                        {"symbol": d["symbol"], "amount": d["dca_dollar_amount"]}
                        for d in decisions
                    ],
                }
            },
        )

        # Execute immediately
        results = execute_paper_decisions(decisions)
        filled = [r for r in results if r.get("status") == "filled"]

        logger.info(
            "Crypto DCA cycle complete",
            extra={
                "extra_data": {
                    "filled": len(filled),
                    "total": len(results),
                    "results": [
                        {"symbol": r["symbol"], "action": r["action"], "status": r["status"]}
                        for r in results
                    ],
                }
            },
        )

    except Exception as e:
        logger.error(
            "Crypto DCA cycle failed",
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
    strategies = config.get("active_strategies", {})
    logger.info(
        "Config loaded",
        extra={
            "extra_data": {
                "active_strategies": strategies,
                "universe_equities": len(config["universe"]["equities"]),
                "universe_crypto": len(config["universe"].get("crypto", [])),
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

    # Job 0a: Weekly deposit (Monday 07:30)
    scheduler.add_job(
        run_weekly_deposit,
        trigger=CronTrigger(day_of_week="mon", hour=7, minute=30, timezone=tz),
        id="weekly_deposit",
        name="Weekly $100 Deposit",
    )

    # Job 0b: Daily performance mode check (08:15)
    scheduler.add_job(
        run_performance_check,
        trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=15, timezone=tz),
        id="performance_check",
        name="Performance Mode Check",
    )

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

    # ── Crypto Jobs (24/7 — every N hours, all 7 days) ─────────────────
    crypto_interval = int(schedule.get("crypto_interval_hours", 2))
    crypto_run_on_startup = schedule.get("crypto_run_on_startup", True)

    # Job 6: Crypto data ingestion (every N hours, 7 days/week)
    scheduler.add_job(
        run_crypto_ingestion,
        trigger=IntervalTrigger(hours=crypto_interval),
        id="crypto_ingestion",
        name=f"Crypto OHLCV Data Ingestion (every {crypto_interval}h, 24/7)",
    )

    # Job 7: Crypto DCA pipeline (every N hours)
    scheduler.add_job(
        run_crypto_dca_cycle,
        trigger=IntervalTrigger(hours=crypto_interval, minutes=5),
        id="crypto_dca_cycle",
        name=f"Crypto DCA Buy Cycle (every {crypto_interval}h, 24/7)",
    )

    jobs = [
        {"id": "weekly_deposit", "trigger": f"Monday at 07:30 {tz}"},
        {"id": "performance_check", "trigger": f"Mon-Fri at 08:15 {tz}"},
        {"id": "daily_ingestion", "trigger": f"Mon-Fri at {schedule['data_ingestion']} {tz}"},
        {"id": "signal_and_decision", "trigger": f"Mon-Fri at {schedule['signal_scoring']} {tz}"},
        {"id": "paper_execution", "trigger": f"Mon-Fri at {schedule['decision_engine']} {tz}"},
        {"id": "eod_sync", "trigger": f"Mon-Fri at {schedule['close_sync']} {tz}"},
        {"id": "weekly_report", "trigger": f"Sunday at 10:00 {tz}"},
        {"id": "crypto_ingestion", "trigger": f"Every {crypto_interval}h (24/7)"},
        {"id": "crypto_dca_cycle", "trigger": f"Every {crypto_interval}h +5min (24/7)"},
    ]

    logger.info(
        "Scheduler configured",
        extra={"extra_data": {"jobs": jobs}},
    )

    logger.info("Starting scheduler — press Ctrl+C to exit")

    # Run first crypto DCA cycle immediately on startup (don't wait for interval)
    if crypto_run_on_startup and config["universe"].get("crypto"):
        logger.info("Running crypto DCA cycle on startup")
        try:
            run_crypto_ingestion()
            run_crypto_dca_cycle()
        except Exception as e:
            logger.error(
                "Startup crypto DCA cycle failed (non-fatal)",
                extra={"extra_data": {"error": str(e)}},
            )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
