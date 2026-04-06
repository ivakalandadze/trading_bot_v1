# Autonomous Trading Bot v1.0

10-technique market analysis with LLM judge, risk management, and paper + live execution.

## How It Works

```
Market Data → 10 Analysis Techniques (parallel) → Vote Consensus → LLM Judge → Risk Manager → Execute
```

1. **Scan** — Fetches price history, fundamentals, earnings, macro data for 40 stocks + 10 crypto
2. **Analyse** — 10 techniques run in parallel, each voting BUY / SELL / NEUTRAL with a score and confidence
3. **Consensus** — If 3+ techniques agree on direction, a trading signal is raised
4. **LLM Gate** — Claude reviews the signal and approves or rejects based on multi-factor quality
5. **Risk Check** — ATR-based stop-loss, position sizing (2% risk per trade), portfolio risk caps, daily loss limit
6. **Execute** — Paper mode simulates trades; live mode routes to Alpaca (stocks) or Binance (crypto)

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run
python main.py              # Interactive menu
python main.py --scan       # One scan cycle
python main.py --symbol AAPL  # Analyse a single stock
python main.py --mode paper   # Start paper trading loop
```

## The 10 Techniques

| # | Name | Focus |
|---|------|-------|
| T1 | Screener | Fundamental quality — P/E, growth, ROE, margins |
| T2 | DCF | Intrinsic value — discounted cash flow vs market cap |
| T3 | Risk | Risk filter — volatility, Sharpe, max drawdown, VaR |
| T4 | Earnings | Earnings momentum — beat rate, EPS growth, PEG |
| T5 | Portfolio | Portfolio fit — sector diversification, correlation, volatility |
| T6 | Technical | Timing — MA crossovers, RSI, MACD, Bollinger, volume |
| T7 | Dividend | Income quality — yield, payout ratio, dividend growth |
| T8 | Competitive | Competitive position — margins vs peers, moat assessment |
| T9 | Patterns | Statistical edge — momentum, seasonality, insider activity |
| T10 | Macro | Macro environment — VIX, rates, DXY, sector rotation |

## Risk Controls

- **2% max risk per trade** (configurable)
- **ATR-based stop-loss** with trailing stops
- **Max 10 open positions** (configurable)
- **6% max portfolio risk** across all positions
- **Daily loss limit** — halts trading at -5% daily P&L
- **Market hours check** — skips stock trades when US market is closed
- **Duplicate trade guard** — prevents opening the same position twice

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `PAPER_CAPITAL` | `10000` | Starting capital ($) |
| `RISK_PER_TRADE` | `0.02` | Max risk per trade (2%) |
| `MAX_OPEN_POSITIONS` | `10` | Concurrent position limit |
| `MIN_SIGNALS_TO_TRADE` | `3` | Techniques needed for consensus |
| `SCAN_INTERVAL` | `60` | Minutes between scans |

## Architecture

```
main.py                    CLI entry point + interactive menu
├── trading_engine.py      Orchestrator: scan → judge → risk → execute
├── signal_engine.py       Runs 10 techniques, aggregates votes
├── llm_judge.py           Claude AI final approval gate
├── risk_manager.py        Position sizing, stops, portfolio limits
├── paper_trader.py        Simulated execution
├── database.py            SQLite persistence
├── config.py              Environment-based configuration
├── data/
│   └── market_data.py     yfinance + Binance data fetching
├── broker/
│   ├── alpaca_broker.py   Live stock execution
│   └── binance_broker.py  Live crypto execution
└── techniques/
    ├── base_technique.py  Shared contract
    └── t1–t10             Individual analysis modules
```
