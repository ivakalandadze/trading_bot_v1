# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # then fill in real credentials

# Run (interactive menu)
python main.py

# Common CLI flags
python main.py --scan              # one scan cycle and exit
python main.py --status            # print status and exit
python main.py --symbol AAPL       # analyse a single symbol
python main.py --mode paper        # force paper trading mode
python main.py --mode live         # force live trading mode
python main.py --interval 30       # override scan interval (minutes)
```

Logs are written to `trading_bot.log` (appended) and stdout simultaneously.

There is no test suite in this repository.

## Architecture

The pipeline is: **Market Data → Signal Engine (10 techniques, parallel) → LLM Judge → Risk Manager → Execute**

### Core modules

| Module | Role |
|--------|------|
| `main.py` | CLI entry, interactive menu, `--scan`/`--status`/`--symbol` flags |
| `trading_engine.py` | Master orchestrator — wires SignalEngine → LLMJudge → RiskManager → execution |
| `signal_engine.py` | Fetches a shared data bundle, fans out to all 10 techniques via `ThreadPoolExecutor`, aggregates weighted votes |
| `llm_judge.py` | Calls Claude (`claude-sonnet-4-6`) with a structured briefing; falls back to auto-approve when `ANTHROPIC_API_KEY` is absent |
| `risk_manager.py` | ATR-based stop-loss, position sizing (2% risk/trade), portfolio caps, daily loss limit |
| `paper_trader.py` | Simulated execution against SQLite state |
| `database.py` | SQLite persistence — trades, positions, signals, scan logs |
| `config.py` | Single source of truth for all settings; loads from `.env` via `python-dotenv` |
| `data/market_data.py` | yfinance (stocks/macro) + Binance REST (crypto) data fetching |
| `broker/alpaca_broker.py` | Live stock execution via Alpaca |
| `broker/binance_broker.py` | Live crypto execution via Binance |

### Techniques (`techniques/`)

All 10 techniques inherit from `BaseTechnique` and implement `analyse(symbol, data) → TechniqueResult`. Each returns `signal` (BUY/SELL/NEUTRAL), `score` (0–100), `confidence` (0–1), and `applicable` (False for techniques that don't apply, e.g., DCF for crypto).

`techniques/__init__.py` exports `ALL_TECHNIQUES` — the ordered list consumed by `SignalEngine`. Add a new technique by subclassing `BaseTechnique` and appending to `ALL_TECHNIQUES`.

The signal data bundle (`_fetch_data_bundle` in `signal_engine.py`) is fetched once per symbol and shared across all techniques. Macro data is cached in-memory for 30 minutes.

Technique weights for the aggregate score are defined in `config.TECHNIQUE_WEIGHTS`. A consensus requires `MIN_SIGNALS_TO_TRADE` (default 3) applicable techniques to agree on direction.

### Data flow detail

1. `SignalEngine.scan_universe()` iterates all symbols in `STOCK_UNIVERSE + CRYPTO_UNIVERSE`
2. Per symbol: fetch data bundle → run 10 techniques concurrently → count votes → if consensus, build `TradingSignal`
3. `TradingEngine._process_signal()`: get current price → `RiskManager.calculate_trade()` → `LLMJudge.judge()` → execute (paper or live)
4. All trades, positions, and individual technique signals are persisted to SQLite

### Key design decisions

- **Only BUY signals are executed** — no short selling in the current implementation (`_process_signal` explicitly skips SELL directions)
- **Crypto detection** uses `is_crypto(symbol)` from `data/market_data.py`, routing to Binance vs. Alpaca accordingly
- **Live brokers are lazily initialised** only when `TRADING_MODE=live`
- All configuration flows through `config.py`; never import `os.getenv` directly in other modules

## Security note

The committed `.env.example` contains real API keys. These should be rotated and the file should be replaced with placeholder values only.
