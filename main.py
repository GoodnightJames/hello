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


def run_crypto_exit_check():
    """
    Check all crypto positions for exit conditions (take-profit, trailing stop, hard stop).

    Runs as part of the crypto DCA cycle. Sells execute first so proceeds
    recycle back to the sleeve before the next buy.

    Returns:
        List of execution results from any sells triggered.
    """
    try:
        from strategies.crypto_dca_v1 import CryptoDCAStrategy
        from data.db import get_session as get_db_session, CostBasis, PositionHighWater
        from execution.alpaca_broker import get_positions_with_retry as get_alpaca_positions

        strategy = CryptoDCAStrategy()
        instruments = set(strategy.get_instruments())

        # Get live positions from Alpaca
        all_positions = get_alpaca_positions()
        crypto_positions = {
            sym: pos for sym, pos in all_positions.items()
            if sym in instruments
        }

        if not crypto_positions:
            logger.info("Crypto exit check: no crypto positions held")
            return []

        # Get cost bases, high-water marks, and entry times from DB
        session = get_db_session()
        try:
            cost_bases = {}
            entry_times = {}
            for sym in crypto_positions:
                basis = session.query(CostBasis).filter(CostBasis.symbol == sym).first()
                if basis and basis.avg_price > 0:
                    cost_bases[sym] = {"avg_price": basis.avg_price, "qty": basis.qty}
                    # Use cost basis updated_at as entry time proxy
                    if hasattr(basis, 'updated_at') and basis.updated_at:
                        entry_times[sym] = basis.updated_at

            high_water_marks = {}
            for sym in crypto_positions:
                hw = session.query(PositionHighWater).filter(PositionHighWater.symbol == sym).first()
                if hw:
                    high_water_marks[sym] = {"high_price": hw.high_price, "entry_price": hw.entry_price}
        finally:
            session.close()

        # Generate exit signals (with time-decay and momentum-collapse checks)
        exit_signals = strategy.generate_exit_signals(
            crypto_positions, cost_bases, high_water_marks,
            entry_times=entry_times,
        )

        if not exit_signals:
            return []

        # Convert exit signals to decisions
        decisions = []
        for sig in exit_signals:
            decisions.append({
                "symbol": sig["symbol"],
                "action": "SELL",
                "reason": sig["reason"],
                "signal_type": "SELL",
                "signal_id": None,
                "position_multiplier": 1.0,
                "risk_approved": True,  # Exits always approved
            })

        logger.info(
            f"Crypto exits: {len(decisions)} sell decision(s)",
            extra={
                "extra_data": {
                    "exits": [{"symbol": d["symbol"], "reason": d["reason"]} for d in decisions],
                }
            },
        )

        # Execute sells
        results = execute_paper_decisions(decisions)
        filled = [r for r in results if r.get("status") == "filled"]

        for r in filled:
            proceeds = r.get("total_proceeds", 0)
            logger.info(
                f"Crypto exit filled: {r['symbol']} → ${proceeds:.2f} back to sleeve",
            )

        return results

    except Exception as e:
        logger.error(
            "Crypto exit check failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        return []


def run_crypto_dca_cycle():
    """
    Scheduled job: crypto DCA cycle — exits first, then buys.

    Runs 24/7, every N hours. Each cycle:
    1. Check all crypto positions for exit conditions (take-profit, stops)
    2. Execute any sells (proceeds recycle to sleeve)
    3. If sleeve has enough cash, buy the next coin in rotation

    This turns the crypto sleeve into a capital recycling engine.
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

        # ── Phase 1: Check exits first ──────────────────────────────
        # Sells recycle capital back to sleeve before we try to buy
        exit_results = run_crypto_exit_check()
        exit_fills = [r for r in exit_results if r.get("status") == "filled"]
        if exit_fills:
            logger.info(f"Crypto exits: {len(exit_fills)} position(s) sold")

        # ── Phase 2: Momentum buy ─────────────────────────────────
        strategy = CryptoDCAStrategy()
        from execution.alpaca_broker import get_positions_with_retry as get_alpaca_positions

        # Check if crypto sleeve has enough capital
        session = get_db_session()
        sleeve_cash = get_sleeve_deployable_cash(session, SLEEVE_CRYPTO)
        session.close()

        total_deploy = strategy.dollars_per_cycle

        if sleeve_cash < total_deploy:
            logger.info(
                f"Crypto DCA: insufficient sleeve cash (${sleeve_cash:.2f} < ${total_deploy:.2f})"
            )
            logger.info("Crypto DCA cycle complete (exits only)")
            return

        # Get currently held symbols so the strategy can deprioritize them
        all_positions = get_alpaca_positions()
        held_symbols = {
            sym for sym in all_positions
            if sym in set(strategy.get_instruments())
        }

        # ── Universe filter ───────────────────────────────────────────
        from risk.universe import filter_tradeable_universe
        all_crypto_symbols = strategy.get_instruments()
        try:
            prices = get_price_history(all_crypto_symbols, lookback_days=30)
            eligible_symbols, removed_symbols = filter_tradeable_universe(
                all_crypto_symbols, prices
            )
        except Exception:
            eligible_symbols = list(all_crypto_symbols)
            removed_symbols = []

        # ── Regime tag ────────────────────────────────────────────────
        from risk.regime_tagger import tag_current_regime
        try:
            features = {"prices": get_price_history(["SPY"], lookback_days=250)}
            regime = tag_current_regime(features)
        except Exception:
            regime = {"trend": "unknown", "vol": "unknown", "phase": "unknown"}

        # ── Allocation policy ─────────────────────────────────────────
        # Enforce symbol culling: dead_weight → disabled,
        # conditional_earner → regime-gated, core_earner → normal.
        # Runs AFTER universe filter, BEFORE scoring.
        from risk.allocation_policy import refresh_and_apply
        try:
            eligible_symbols, policy_blocked, alloc_policy = refresh_and_apply(
                eligible_symbols, regime=regime,
            )
            if policy_blocked:
                logger.info(
                    f"Allocation policy blocked {len(policy_blocked)} symbol(s)",
                    extra={"extra_data": {
                        "blocked": [(s, r) for s, r in policy_blocked],
                    }},
                )
        except Exception as e:
            logger.warning(
                f"Allocation policy check failed (non-fatal): {e}",
            )
            policy_blocked = []
            alloc_policy = {}

        # ── Risk budget ──────────────────────────────────────────────
        from risk.sleeve_risk import check_sleeve_risk_budget
        from risk.enforcer import load_risk_params
        risk_params = load_risk_params()
        budget_result = check_sleeve_risk_budget(session, risk_params)
        budget_scale = budget_result.get("position_scale", 1.0)

        # ── Sleeve health check ─────────────────────────────────────
        # Meta-layer: reduce aggressiveness when the sleeve is "out of form."
        from risk.sleeve_health import check_sleeve_health
        health_session = get_db_session()
        health = check_sleeve_health(health_session, sleeve="crypto")
        health_session.close()
        health_mult = health.get("aggressiveness", 1.0)

        if health_mult < 1.0:
            logger.info(
                f"Sleeve health: score={health['score']:.2f} → "
                f"aggressiveness={health_mult:.2f} ({health['recommendation']})"
            )

        # Generate buy signal — picks best coin by risk-adjusted momentum.
        # Passes regime so thresholds adapt to market conditions.
        scored = strategy._score_coins(eligible_symbols, held_symbols)
        signals = strategy.generate_signals(
            held_symbols=held_symbols, regime=regime
        )

        # ── Shadow mode: score blocked symbols hypothetically ─────────
        # Track what would have happened without policy enforcement.
        if policy_blocked:
            from risk.allocation_policy import record_shadow_outcome
            blocked_syms = [s for s, _ in policy_blocked]
            try:
                shadow_scored = strategy._score_coins(blocked_syms, held_symbols)
                best_live = scored[0][1] if scored else -999
                for sym, shadow_score, _, shadow_diag in shadow_scored:
                    # Would this blocked symbol have been selected?
                    would_selected = (
                        shadow_score > best_live
                        and shadow_score >= strategy.min_score_threshold
                        and shadow_diag.get("passes_cost_gate", False)
                    )
                    block_reason = dict(policy_blocked).get(sym, "")
                    action = "disable" if "DISABLED" in block_reason else "restrict_to_regimes"
                    record_shadow_outcome(
                        symbol=sym,
                        action=action,
                        regime_phase=regime.get("phase", "unknown") if regime else "unknown",
                        would_have_scored=shadow_score,
                        would_have_selected=would_selected,
                    )
            except Exception:
                pass  # Shadow scoring is non-critical

        # ── Instrument the cycle ──────────────────────────────────────
        from review.cycle_instrumentation import instrument_crypto_cycle
        pre_notional = total_deploy
        post_notional = total_deploy * budget_scale
        is_crypto_min = 10.0
        notional_pass = post_notional >= is_crypto_min

        instrument_crypto_cycle(
            scored=scored,
            signals=signals,
            eligible_symbols=eligible_symbols,
            removed_symbols=removed_symbols,
            all_symbols=all_crypto_symbols,
            budget_result=budget_result,
            regime=regime,
            pre_scale_notional=pre_notional,
            post_scale_notional=post_notional,
            min_notional_pass=notional_pass,
            min_score_threshold=strategy.min_score_threshold,
            policy_blocked=policy_blocked,
        )

        if not signals:
            logger.info("Crypto DCA: no signals generated (threshold or cost gate)")
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
                    "exits_filled": len(exit_fills),
                    "buys_filled": len(filled),
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

    # ── Crypto Jobs (24/7 — all 7 days) ────────────────────────────────
    crypto_buy_interval = int(schedule.get("crypto_buy_interval_hours", schedule.get("crypto_interval_hours", 2)))
    crypto_exit_interval = int(schedule.get("crypto_exit_interval_minutes", 15))
    crypto_run_on_startup = schedule.get("crypto_run_on_startup", True)

    # Job 6: Crypto data ingestion (every 2 hours, 7 days/week)
    scheduler.add_job(
        run_crypto_ingestion,
        trigger=IntervalTrigger(hours=crypto_buy_interval),
        id="crypto_ingestion",
        name=f"Crypto OHLCV Data Ingestion (every {crypto_buy_interval}h, 24/7)",
    )

    # Job 7: Crypto BUY cycle (every 2 hours — rotation buys + exit check)
    scheduler.add_job(
        run_crypto_dca_cycle,
        trigger=IntervalTrigger(hours=crypto_buy_interval, minutes=5),
        id="crypto_dca_cycle",
        name=f"Crypto Buy Cycle (every {crypto_buy_interval}h, 24/7)",
    )

    # Job 8: Crypto EXIT check (every 15 minutes — fast capture)
    # This is the key speed advantage: catches take-profit/stop-loss
    # opportunities within minutes, not hours.
    scheduler.add_job(
        run_crypto_exit_check,
        trigger=IntervalTrigger(minutes=crypto_exit_interval),
        id="crypto_exit_check",
        name=f"Crypto Exit Check (every {crypto_exit_interval}min, 24/7)",
    )

    jobs = [
        {"id": "weekly_deposit", "trigger": f"Monday at 07:30 {tz}"},
        {"id": "performance_check", "trigger": f"Mon-Fri at 08:15 {tz}"},
        {"id": "daily_ingestion", "trigger": f"Mon-Fri at {schedule['data_ingestion']} {tz}"},
        {"id": "signal_and_decision", "trigger": f"Mon-Fri at {schedule['signal_scoring']} {tz}"},
        {"id": "paper_execution", "trigger": f"Mon-Fri at {schedule['decision_engine']} {tz}"},
        {"id": "eod_sync", "trigger": f"Mon-Fri at {schedule['close_sync']} {tz}"},
        {"id": "weekly_report", "trigger": f"Sunday at 10:00 {tz}"},
        {"id": "crypto_ingestion", "trigger": f"Every {crypto_buy_interval}h (24/7)"},
        {"id": "crypto_dca_cycle", "trigger": f"Every {crypto_buy_interval}h +5min (24/7)"},
        {"id": "crypto_exit_check", "trigger": f"Every {crypto_exit_interval}min (24/7)"},
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
