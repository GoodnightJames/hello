"""
Portfolio status dashboard — see what you own at a glance.

Usage:
    python scripts/status.py
"""

import sys
sys.path.insert(0, ".")

from data.db import init_db, get_session, CostBasis, Trade, Deposit, Order
from capital.manager import get_sleeve_summary, SLEEVE_EQUITY, SLEEVE_CRYPTO
from execution.alpaca_broker import get_positions, get_account


def main():
    init_db()
    session = get_session()

    # ── Alpaca account ────────────────────────────────────────
    account = get_account()
    positions = get_positions()

    # ── Sleeve balances ───────────────────────────────────────
    sleeve = get_sleeve_summary(session)
    equity_sleeve = sleeve.get(SLEEVE_EQUITY, {})
    crypto_sleeve = sleeve.get(SLEEVE_CRYPTO, {})

    # ── Deposits ──────────────────────────────────────────────
    deposits = session.query(Deposit).order_by(Deposit.date.desc()).all()
    total_deposited = sum(d.amount for d in deposits)

    # ── Realized trades ───────────────────────────────────────
    trades = session.query(Trade).all()
    total_realized_pnl = sum(t.realized_pnl for t in trades) if trades else 0

    # ── Open orders ───────────────────────────────────────────
    open_orders = session.query(Order).filter(
        Order.status.notin_(["filled", "canceled", "expired", "rejected"])
    ).all()

    # ── Cost bases ────────────────────────────────────────────
    cost_bases = {cb.symbol: cb for cb in session.query(CostBasis).all() if cb.qty > 0}

    session.close()

    # ══════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════

    print()
    print("=" * 60)
    print("  PORTFOLIO DASHBOARD")
    print("=" * 60)

    # Account summary
    print(f"\n  Alpaca Account")
    print(f"  ├─ Cash:          ${account['cash']:>10,.2f}")
    print(f"  ├─ Portfolio:     ${account['equity']:>10,.2f}")
    print(f"  └─ Status:        {account['status']}")

    # Deposits
    print(f"\n  Deposits")
    print(f"  ├─ Total in:      ${total_deposited:>10,.2f}")
    print(f"  └─ # deposits:    {len(deposits)}")

    # Sleeves
    print(f"\n  Sleeves (virtual cash buckets)")
    print(f"  ├─ Equity:        ${equity_sleeve.get('cash', 0):>10,.2f}")
    print(f"  └─ Crypto:        ${crypto_sleeve.get('cash', 0):>10,.2f}")

    # Positions
    print(f"\n  Open Positions ({len(positions)})")
    if positions:
        print(f"  {'Symbol':<12} {'Qty':>12} {'Entry':>10} {'Now':>10} {'Value':>10} {'P&L':>10} {'%':>7}")
        print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*7}")

        total_value = 0
        total_unrealized = 0

        for symbol, pos in sorted(positions.items()):
            qty = pos["qty"]
            current = pos["current_price"]
            value = pos["market_value"]
            entry = pos["avg_entry"]
            pnl = pos["unrealized_pl"]
            pct = ((current - entry) / entry * 100) if entry > 0 else 0

            pnl_str = f"{'+'if pnl >= 0 else ''}{pnl:.2f}"
            pct_str = f"{'+'if pct >= 0 else ''}{pct:.1f}%"

            print(f"  {symbol:<12} {qty:>12.6f} ${entry:>9.4f} ${current:>9.4f} ${value:>9.2f} {pnl_str:>10} {pct_str:>7}")

            total_value += value
            total_unrealized += pnl

        print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*7}")
        unrealized_str = f"{'+'if total_unrealized >= 0 else ''}{total_unrealized:.2f}"
        print(f"  {'TOTAL':<12} {'':>12} {'':>10} {'':>10} ${total_value:>9.2f} {unrealized_str:>10}")
    else:
        print("  (none)")

    # Realized P&L
    print(f"\n  Realized P&L")
    if trades:
        print(f"  ├─ # trades:      {len(trades)}")
        print(f"  └─ Total P&L:     ${'+'if total_realized_pnl >= 0 else ''}{total_realized_pnl:>9,.2f}")
    else:
        print(f"  └─ No completed trades yet")

    # Overall
    total_position_value = sum(p["market_value"] for p in positions.values())
    total_assets = account["cash"] + total_position_value
    overall_pnl = total_assets - total_deposited if total_deposited > 0 else 0

    print(f"\n  Overall")
    print(f"  ├─ Total assets:  ${total_assets:>10,.2f}")
    print(f"  ├─ Total in:      ${total_deposited:>10,.2f}")
    pnl_sign = '+' if overall_pnl >= 0 else ''
    print(f"  └─ Net P&L:       ${pnl_sign}{overall_pnl:>9,.2f}")

    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
