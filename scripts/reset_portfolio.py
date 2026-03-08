"""
Reset portfolio — sell all positions, wipe state, start fresh.

Usage:
    python scripts/reset_portfolio.py          # Reset with $100 deposit
    python scripts/reset_portfolio.py 200      # Reset with $200 deposit
"""

import sys
import time

sys.path.insert(0, ".")

from data.db import (
    init_db, get_session, get_engine, Base,
    CostBasis, PositionHighWater, SleeveBalance, PortfolioState,
    Deposit, Order, Signal, Decision, Trade, RiskEvent,
)
from capital.manager import record_deposit
from risk.enforcer import load_risk_params
from performance.manager import select_risk_mode, get_mode_params
from execution.alpaca_broker import get_positions, submit_market_order, get_client


def sell_all_positions():
    """Sell every open position on Alpaca."""
    positions = get_positions()
    if not positions:
        print("  No open positions to sell.")
        return

    results = []
    for symbol, pos in positions.items():
        qty = pos["qty"]
        if qty <= 0:
            continue

        print(f"  Selling {qty} {symbol} (value ~${pos['market_value']:.2f})...")

        # Alpaca needs the raw symbol for crypto (e.g., BTC/USD)
        try:
            order = submit_market_order(symbol, qty, "sell")
            # Wait for fill
            client = get_client()
            for _ in range(15):
                time.sleep(2)
                status = client.get_order_by_id(order["broker_order_id"])
                if hasattr(status.status, 'value'):
                    st = status.status.value
                else:
                    st = str(status.status)
                if st == "filled":
                    filled_price = float(status.filled_avg_price) if status.filled_avg_price else 0
                    print(f"    SOLD {qty} {symbol} @ ${filled_price:.6f}")
                    results.append({"symbol": symbol, "qty": qty, "price": filled_price})
                    break
            else:
                print(f"    WARNING: {symbol} sell order not filled within 30s")
        except Exception as e:
            print(f"    ERROR selling {symbol}: {e}")

    return results


def wipe_database():
    """Clear all portfolio state tables (keeps DailyBar price history)."""
    engine = get_engine()
    session = get_session(engine)

    tables_to_clear = [
        CostBasis,
        PositionHighWater,
        SleeveBalance,
        PortfolioState,
        Deposit,
        Order,
        Signal,
        Decision,
        Trade,
        RiskEvent,
    ]

    for model in tables_to_clear:
        count = session.query(model).count()
        session.query(model).delete()
        print(f"  Cleared {model.__tablename__}: {count} rows")

    session.commit()
    session.close()


def main():
    deposit_amount = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

    print("=" * 50)
    print("PORTFOLIO RESET")
    print("=" * 50)

    # Step 1: Initialize DB connection
    init_db()

    # Step 2: Sell all positions on Alpaca
    print("\n[1/3] Selling all positions...")
    sell_all_positions()

    # Step 3: Wipe database state
    print("\n[2/3] Wiping database state...")
    wipe_database()

    # Step 4: Record fresh deposit
    print(f"\n[3/3] Recording fresh ${deposit_amount:.2f} deposit...")
    session = get_session()
    risk_params = load_risk_params()
    mode_result = select_risk_mode(session, risk_params)
    mode_params = get_mode_params(risk_params, mode_result["mode"])

    portfolio = record_deposit(
        session,
        amount=deposit_amount,
        notes="Fresh start — portfolio reset",
        mode_params=mode_params,
    )
    session.commit()

    # Show result
    from capital.manager import get_sleeve_summary
    summary = get_sleeve_summary(session)
    session.close()

    print("\n" + "=" * 50)
    print("RESET COMPLETE")
    print("=" * 50)
    print(f"  Deposited:      ${deposit_amount:.2f}")
    print(f"  Equity sleeve:  ${summary['equity']['cash']:.2f}")
    print(f"  Crypto sleeve:  ${summary['crypto']['cash']:.2f}")
    print(f"  Total equity:   ${portfolio['total_equity']:.2f}")
    print("\nReady to start fresh. Run: caffeinate -s python main.py")


if __name__ == "__main__":
    main()
