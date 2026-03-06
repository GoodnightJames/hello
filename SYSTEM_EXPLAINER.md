# What Is This? — An Algorithmic Trading System

This document explains the same system at three levels. Pick the one that matches your comfort with tech.

**The plan:** Start from $0. Put in $100 every week. Let the system invest it automatically. Get rich or go broke.

---

## Level 1: For Anyone (No Tech Background)

### What does it do?

Every week, $100 goes into a brokerage account. This software automatically decides what to buy with that $100 — no human decision-making involved. It checks prices every day and follows strict rules about what to invest in.

### How does it pick what to buy?

It follows a well-known strategy called **Dual Momentum**, published by Gary Antonacci in 2014. The idea is simple:

1. **Is the stock market going up overall?** If yes, invest the $100 into stocks. If no, put it in safe bonds until things improve.
2. **Which stocks are going up the fastest?** Buy the ones with the strongest upward trend.

That's it. No AI guessing. No "hot tips." Just a disciplined, mechanical process that checks prices once a day and invests each week's deposit into whatever has the best momentum.

### How much money are we talking about?

- **Week 1:** $100 invested
- **Month 1:** ~$400 invested
- **Year 1:** ~$5,200 invested
- **Year 5:** ~$26,000 deposited — but if the strategy works, the investments grow too

The magic is **compounding** — early gains get reinvested alongside new deposits, so the pile grows faster over time.

### What keeps it safe?

The system has safety nets, but they're calibrated for a small account:

- **Daily loss limit** — If it loses more than 5% in a single day, it stops trading for the rest of the day. On a $500 account, that's $25 — not catastrophic.
- **Weekly loss limit** — If losses hit 10% in a week, it enters review mode.
- **Emergency off switch** — We can shut it all down instantly with one command.
- **Paper trading first** — It runs with simulated money for at least 3 weeks to prove it works before real money is used.

### What's the goal?

Build a system that invests $100/week with discipline no human can match — no fear, no greed, no "I have a feeling about this stock." Just rules, executed consistently, every single week.

---

## Level 2: For People Comfortable with Finance and Basic Tech

### Architecture at a Glance

The system runs as a Python application on a scheduled daily loop. It connects to the Alpaca brokerage API for paper trading (simulated trades with real market prices). It's built around a **$100/week deposit cadence** — every dollar gets deployed.

**Daily pipeline (Mon–Fri):**

| Time (ET) | Step | What happens |
|-----------|------|--------------|
| 8:00 AM | Data Ingestion | Pulls 1 year of daily price data (open, high, low, close, volume) for 31 instruments |
| 8:30 AM | Signal Generation | Computes 12-month momentum scores, classifies market regime, generates BUY/SELL/HOLD signals |
| 9:25 AM | Execution | Validates signals against risk limits, deploys available cash into best signal |
| 4:00 PM | End-of-Day Sync | Reconciles portfolio with broker, records daily snapshot |
| Sunday 10 AM | Weekly Report | Summarizes performance, flags risk events |

### Strategy: Dual Momentum + DCA

The core strategy is DCA (dollar-cost averaging) $100/week with a momentum tilt. Instead of blindly buying SPY every week, the system picks the best asset based on momentum.

**Decision logic:**

1. Compute 12-month return for each risk asset (SPY, QQQ, etc.)
2. Compare each to SHY (short-term treasuries) — this is the **absolute momentum** check
3. If an asset's return beats SHY, it has positive momentum
4. Rank all passing assets by return — the top one gets the weekly deposit (**relative momentum**)
5. If nothing beats SHY, rotate entirely to bonds (AGG/SHY) — the money still gets deployed, just into safety

**Regime filter overlay:**
- If SPY is below its 200-day moving average → risk-off, no new equity buys
- If VIX is above 30 → reduce position sizes by 50%
- Both conditions active → reduce to 25%

### Risk Management (Tuned for Small Accounts)

Traditional risk limits (1% per trade) don't work on $400. The system uses **cash deployment percentages** instead:

| Mode | When it activates | Cash deployed per trade | Max positions |
|------|-------------------|------------------------|---------------|
| Conservative | Drawdown ≥ 10% or 5+ consecutive losses | 50% of cash | 2 |
| Normal | Default state | 90% of cash | 3 |
| Aggressive | 5%+ returns with <6% drawdown | 95% of cash | 4 |

Hard limits:
- 5% max daily loss → automatic shutdown
- 10% max weekly drawdown → review mode
- 5 consecutive losses → halt
- 6 trades/week maximum
- Kill switch for emergency stop

These are deliberately wider than institutional limits because a 10% drawdown on $500 is $50 — two bad lunches, not a crisis.

### Go-Live Criteria

The system must pass ALL of these in paper trading before real money:
- 21+ days of operation (3 weeks)
- 6+ trades completed
- Max drawdown under 15%
- Win rate above 35% (momentum wins big, not often)
- Profit factor ≥ 1.0 (gross profits exceed gross losses)

### Capital Model

- **Starting capital: $0** (no lump sum)
- **Weekly deposit: $100** (this IS the strategy)
- **Allocation: 95%** to strategy, 5% cash buffer
- **Key metric: total deposited vs current value** — that's the only number that matters

---

## Level 3: For Engineers and Developers

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Broker API | alpaca-py (paper trading) |
| Alt data source | yfinance |
| Database | SQLite via SQLAlchemy ORM |
| Scheduler | APScheduler (cron triggers) |
| Technical indicators | pandas-ta |
| Configuration | YAML (settings, risk params, strategy configs) |
| Logging | Structured JSON (one file per day) |

### Module Architecture

```
main.py                    # APScheduler entry point, 5 cron jobs
│
├── data/
│   ├── db.py              # SQLAlchemy models: DailyBar, Signal, Decision,
│   │                      #   Order, RiskEvent, PortfolioState, ParamVersion, Deposit
│   ├── ingestion.py       # Alpaca historical bar fetcher (365-day rolling window)
│   └── feature_store.py   # Computes returns (1m/3m/6m/12m), SMA(50/200), RSI(2)
│
├── research/
│   ├── momentum_scorer.py # Absolute + relative momentum scoring vs SHY benchmark
│   └── regime.py          # SPY trend (200d SMA) + VIX vol classification
│
├── strategies/
│   ├── base.py            # Abstract StrategyBase class
│   └── momentum_v1.py     # Dual Momentum implementation (Antonacci 2014)
│
├── decision/
│   └── engine.py          # Signal → Decision conversion with regime filter
│
├── risk/
│   └── enforcer.py        # Pre-trade risk validation (kill switch, limits, throttle)
│
├── execution/
│   ├── alpaca_broker.py   # Alpaca API wrapper (submit, cancel, get positions)
│   ├── paper.py           # Paper execution engine (order → fill → portfolio update)
│   └── order_manager.py   # Order lifecycle management
│
├── capital/
│   └── manager.py         # Portfolio state tracking, deposit handling
│
├── performance/
│   └── manager.py         # Rolling equity analysis, dynamic risk mode selection
│
├── review/
│   ├── weekly_report.py   # Performance summary generation
│   └── param_log.py       # Strategy parameter versioning (audit trail)
│
├── replay/
│   └── engine.py          # Backtesting scaffold (future)
│
├── config/
│   ├── settings.yaml      # Universe (31 symbols), schedule, rebalance frequency
│   ├── risk_params.yaml   # All risk limits, mode thresholds, go-live gates
│   └── strategies/
│       └── momentum_v1.yaml  # Strategy-specific params
│
└── tests/                 # 7 test modules covering core pipeline
```

### Data Flow

```
Alpaca API  ─→  ingestion.py  ─→  SQLite (DailyBar table)
                                        │
                                  feature_store.py
                                   ├─ returns (1m–12m)
                                   ├─ SMA (50d, 200d)
                                   └─ RSI(2)
                                        │
                                  momentum_scorer.py
                                   ├─ absolute check (vs SHY)
                                   └─ relative ranking
                                        │
                                  regime.py
                                   ├─ trend: SPY vs 200d SMA
                                   └─ vol: VIX threshold
                                        │
                                  decision/engine.py
                                   ├─ regime filter applied
                                   └─ BUY / SELL / HOLD / SKIP
                                        │
                                  risk/enforcer.py
                                   ├─ kill switch
                                   ├─ daily/weekly loss limits
                                   ├─ position limits (mode-aware)
                                   ├─ trade throttle
                                   └─ approved / rejected
                                        │
                                  execution/paper.py
                                   ├─ calculate shares from $ allocation
                                   ├─ submit market order to Alpaca
                                   ├─ poll fill (1s interval, 30s max)
                                   └─ update portfolio state
```

### Design Principles

1. **Strict separation of concerns** — The decision engine knows nothing about risk limits. The risk enforcer knows nothing about strategy logic. Execution knows nothing about why a trade was chosen. Each layer is independently testable.

2. **Determinism** — Same inputs always produce the same signals and decisions. Every decision is timestamped and logged to the database for full auditability.

3. **Fail-safe defaults** — Unknown regime → assume risk-off. Missing data → skip execution. API error → retry with exponential backoff. Any doubt → don't trade.

4. **Sell orders always approved** — Risk enforcement only gates new positions. Reducing exposure is always allowed.

5. **Mode transition dampening** — Minimum 10 days in any risk mode before transitioning. Prevents whipsawing between conservative and aggressive on noisy equity curves.

6. **Local state as source of truth** — SQLite portfolio state is authoritative. Alpaca positions are used for reconciliation, not as the primary record.

### Database Schema (8 tables)

- `DailyBar` — OHLCV price data per symbol per day
- `Signal` — Raw strategy outputs (symbol, direction, score, reason)
- `Decision` — Filtered signals after regime overlay (action + reasoning)
- `Order` — Full order lifecycle (pending → submitted → filled/rejected)
- `RiskEvent` — Every risk check result (shutdowns, throttles, mode changes)
- `PortfolioState` — Daily snapshots (cash, equity, positions as JSON)
- `ParamVersion` — Strategy config change log
- `Deposit` — Capital inflow tracking

### Key Config Changes for Accumulation Mode

The system was originally designed for a $100k paper trading account. Here's what changed:

| Parameter | Old ($100k) | New ($100/wk) | Why |
|-----------|-------------|---------------|-----|
| `initial_capital` | $100,000 | $0 | No lump sum |
| `allocation.capital_pct` | 60% | 95% | Can't afford idle cash |
| `risk_per_trade_pct` | 1% of equity | 90% of cash | Deploy the deposit |
| `max_daily_loss` | 2% | 5% | $10 on $200 is noise |
| `max_weekly_drawdown` | 5% | 10% | Same reason |
| `consecutive_loss_shutdown` | 3 | 5 | More room on small $ |
| `rebalance` | Monthly | Weekly | Align with deposit cadence |
| `go_live min_days` | 30 | 21 | Faster feedback loop |

### Running It

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env   # Add Alpaca API keys

# Run the scheduler (stays alive, fires jobs on cron schedule)
python main.py

# Check system status (shows deposited vs current value)
python status.py
```

### What's Not Built Yet

- `replay/engine.py` is scaffolded but not implemented (backtesting)
- VIX data integration defaults to "normal" volatility (no live VIX feed yet)
- Only one strategy slot active (system supports multiple)
- No intraday data — daily bars only
- Consecutive loss tracking uses order count, not realized P&L

---

*Document generated from the codebase. Account model: $0 start, $100/week deposits, get rich or go broke. For questions, review the config files in `config/` — they're heavily commented and serve as the system's specification.*
