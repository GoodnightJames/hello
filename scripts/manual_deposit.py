"""Manual deposit — inject capital outside the weekly schedule."""
import sys
sys.path.insert(0, ".")

from data.db import init_db, get_session
from capital.manager import record_deposit, get_sleeve_summary
from risk.enforcer import load_risk_params
from performance.manager import select_risk_mode, get_mode_params


def main():
    amount = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

    init_db()
    session = get_session()

    risk_params = load_risk_params()
    mode_result = select_risk_mode(session, risk_params)
    mode_params = get_mode_params(risk_params, mode_result["mode"])

    portfolio = record_deposit(session, amount=amount, notes="Manual deposit", mode_params=mode_params)
    session.commit()

    summary = get_sleeve_summary(session)
    session.close()

    print(f"Deposited ${amount:.2f}")
    print(f"  Equity sleeve: ${summary['equity']['cash']:.2f}")
    print(f"  Crypto sleeve: ${summary['crypto']['cash']:.2f}")
    print(f"  Total equity:  ${portfolio['total_equity']:.2f}")


if __name__ == "__main__":
    main()
