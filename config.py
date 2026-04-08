"""
config.py — Central configuration loaded from environment variables.
All other modules import from here; never hardcode credentials elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Trading Mode ──────────────────────────────────────────────────────────────
TRADING_MODE: str = os.getenv("TRADING_MODE", "paper").lower()   # 'paper' | 'live'
assert TRADING_MODE in ("paper", "live"), "TRADING_MODE must be 'paper' or 'live'"

# ── Capital & Risk ────────────────────────────────────────────────────────────
PAPER_CAPITAL: float        = float(os.getenv("PAPER_CAPITAL", "10000"))
RISK_PER_TRADE: float       = float(os.getenv("RISK_PER_TRADE", "0.02"))   # 2%
MAX_OPEN_POSITIONS: int     = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
MIN_SIGNALS_TO_TRADE: int   = int(os.getenv("MIN_SIGNALS_TO_TRADE", "6"))  # any 6/10
STOP_LOSS_MULTIPLIER: float = 2.0    # stop-loss = entry - (risk $ / shares) * multiplier
TAKE_PROFIT_RATIO: float    = 2.0    # take-profit = entry + (risk $) * this ratio

# ── Scan Settings ─────────────────────────────────────────────────────────────
SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "60"))   # minutes

# ── Alpaca ────────────────────────────────────────────────────────────────────
ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str         = "claude-sonnet-4-6"

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_PATH: str = os.getenv("DATABASE_PATH", "trading_bot.db")

# ── Stock Universe ────────────────────────────────────────────────────────────
# S&P 500 large-cap sample — expand as needed
STOCK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "JNJ",  "V",    "UNH",  "HD",   "PG",   "MA",   "BAC",  "ABBV",
    "CVX",  "LLY",  "MRK",  "PFE",  "COST", "TMO",  "AVGO", "MCD",
    "CSCO", "ACN",  "NEE",  "DHR",  "TXN",  "VZ",   "AMD",  "ADBE",
    "CRM",  "NFLX", "INTC", "WMT",  "DIS",  "PYPL", "SBUX", "IBM",
]

# ── Crypto Universe ───────────────────────────────────────────────────────────
# Disabled: the analysis techniques are stock-focused (DCF, Earnings, Dividend,
# Competitive, Portfolio all either don't apply or give generic scores for crypto).
# Only T6_Technical and T9_Patterns give crypto genuine analysis, which is
# insufficient signal quality. Re-enable if dedicated crypto techniques are added.
CRYPTO_UNIVERSE: list[str] = []

# ── Macro Tickers (via yfinance) ──────────────────────────────────────────────
MACRO_TICKERS = {
    "vix":        "^VIX",       # Fear index
    "treasury10": "^TNX",       # 10-yr Treasury yield
    "dxy":        "DX-Y.NYB",   # US Dollar index
    "spy":        "SPY",        # S&P 500 ETF (market trend)
    "gold":       "GLD",        # Gold (risk-off indicator)
}

# ── Technique weights for scoring (used by signal engine) ─────────────────────
TECHNIQUE_WEIGHTS: dict[str, float] = {
    "T1_Screener":    1.2,   # Goldman Sachs — fundamental quality
    "T2_DCF":         1.3,   # Morgan Stanley — intrinsic value
    "T3_Risk":        0.8,   # Bridgewater    — risk filter
    "T4_Earnings":    1.1,   # JPMorgan       — earnings momentum
    "T5_Portfolio":   0.7,   # BlackRock      — portfolio fit
    "T6_Technical":   1.2,   # Citadel        — timing
    "T7_Dividend":    0.9,   # Harvard        — income quality
    "T8_Competitive": 1.0,   # Bain           — competitive position
    "T9_Patterns":    1.0,   # Renaissance    — statistical edge
    "T10_Macro":      1.1,   # McKinsey       — macro tailwind
}
