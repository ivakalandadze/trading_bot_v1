"""
data/market_data.py — Unified market data fetcher.

Provides:
  - OHLCV price history (stocks via yfinance, crypto via Binance)
  - Fundamental data (stocks via yfinance)
  - Macro indicators (VIX, 10yr, DXY via yfinance)
  - Current price quotes
  - Asset type detection utilities
"""

import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def _retry(max_attempts: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """Decorator: retry with exponential backoff on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    logger.debug(f"{func.__name__} attempt {attempt} failed: {e} — retrying in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Asset type helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_crypto(symbol: str) -> bool:
    """Detect if symbol is a crypto pair (e.g. BTCUSDT, ETH/USDT)."""
    return (symbol.endswith("USDT") or symbol.endswith("BTC")
            or "/" in symbol or symbol.upper() in
            {"BTC", "ETH", "BNB", "SOL", "ADA", "XRP"})


def crypto_to_yf(symbol: str) -> str:
    """Convert Binance symbol to Yahoo Finance format (BTCUSDT → BTC-USD)."""
    sym = symbol.replace("/", "").replace("USDT", "").replace("USD", "")
    return f"{sym}-USD"


def normalize_symbol(symbol: str) -> str:
    """Return canonical display symbol."""
    if is_crypto(symbol):
        return symbol.replace("USDT", "/USDT")
    return symbol.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Price history
# ─────────────────────────────────────────────────────────────────────────────

@_retry(max_attempts=3, base_delay=0.5)
def _fetch_yf_history(yf_sym: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(yf_sym)
    return ticker.history(period=period, interval=interval, auto_adjust=True)


def get_price_history(symbol: str, period: str = "1y",
                      interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV DataFrame for stocks or crypto.
    Returns columns: Open, High, Low, Close, Volume
    """
    try:
        yf_sym = crypto_to_yf(symbol) if is_crypto(symbol) else symbol
        df = _fetch_yf_history(yf_sym, period, interval)
        if df.empty:
            logger.warning(f"No price history for {symbol}")
            return None
        df.index = pd.to_datetime(df.index)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].dropna()
        return df
    except Exception as e:
        logger.error(f"Price history error for {symbol}: {e}")
        return None


@_retry(max_attempts=3, base_delay=0.5)
def _fetch_current_price(symbol: str) -> Optional[float]:
    if is_crypto(symbol) and config.BINANCE_API_KEY:
        return _binance_price(symbol)
    yf_sym = crypto_to_yf(symbol) if is_crypto(symbol) else symbol
    ticker = yf.Ticker(yf_sym)
    info   = ticker.fast_info
    price  = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
    if price is None:
        hist  = ticker.history(period="2d")
        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
    return float(price) if price else None


def get_current_price(symbol: str) -> Optional[float]:
    """Get latest closing/last price with retry."""
    try:
        return _fetch_current_price(symbol)
    except Exception as e:
        logger.error(f"Price fetch error for {symbol}: {e}")
        return None


def _binance_price(symbol: str) -> Optional[float]:
    """Direct Binance REST call for crypto price."""
    try:
        import requests
        sym = symbol.replace("/", "").upper()
        r = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={sym}",
            timeout=5
        )
        return float(r.json()["price"])
    except Exception as e:
        logger.error(f"Binance price error {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental data (stocks only)
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """
    Fetch key fundamental metrics from yfinance.
    Returns an empty dict for crypto (fundamentals not applicable).
    """
    if is_crypto(symbol):
        return {}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        fin = ticker.financials          # annual P&L — rows=metrics, cols=dates
        cf  = ticker.cashflow            # annual cash flow
        bs  = ticker.balance_sheet       # annual balance sheet

        # ── Revenue growth (5yr) ──────────────────────────────────────────────
        rev_growth = _compute_revenue_growth(fin)

        # ── Free Cash Flow (TTM) ──────────────────────────────────────────────
        fcf_ttm = _compute_fcf(cf)

        # ── Debt to Equity ────────────────────────────────────────────────────
        total_debt   = _safe_get(bs, "Total Debt", 0)
        total_equity = _safe_get(bs, "Stockholders Equity", 1)
        debt_equity  = total_debt / max(total_equity, 1)

        return {
            # Valuation
            "pe_ratio":         info.get("trailingPE"),
            "forward_pe":       info.get("forwardPE"),
            "peg_ratio":        info.get("pegRatio"),
            "price_to_book":    info.get("priceToBook"),
            "ev_to_ebitda":     info.get("enterpriseToEbitda"),
            # Growth & profitability
            "revenue_growth":   info.get("revenueGrowth"),   # YoY TTM
            "revenue_growth_5y": rev_growth,
            "earnings_growth":  info.get("earningsGrowth"),
            "profit_margin":    info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            # Balance sheet
            "debt_to_equity":   info.get("debtToEquity", debt_equity * 100) / 100
                                     if info.get("debtToEquity") else debt_equity,
            "current_ratio":    info.get("currentRatio"),
            "quick_ratio":      info.get("quickRatio"),
            # Dividends
            "dividend_yield":        info.get("dividendYield"),
            "payout_ratio":          info.get("payoutRatio"),
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            # Market data
            "market_cap":       info.get("marketCap"),
            "beta":             info.get("beta"),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            # Company profile
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
            "short_ratio":      info.get("shortRatio"),
            "shares_short_pct": info.get("shortPercentOfFloat"),
            # Cash flow
            "free_cash_flow_ttm": fcf_ttm,
            "operating_cash_flow": info.get("operatingCashflow"),
        }
    except Exception as e:
        logger.error(f"Fundamentals error for {symbol}: {e}")
        return {}


def _compute_revenue_growth(fin: pd.DataFrame) -> Optional[float]:
    """Annualised 5yr revenue CAGR from financial statements."""
    try:
        if fin is None or fin.empty:
            return None
        row = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
        if row is None:
            return None
        values = row.dropna().sort_index().values
        if len(values) < 2:
            return None
        earliest, latest = float(values[0]), float(values[-1])
        n_years = len(values) - 1
        if earliest <= 0 or latest <= 0:
            return None
        cagr = (latest / earliest) ** (1 / n_years) - 1
        return round(cagr, 4)
    except Exception:
        return None


def _compute_fcf(cf: pd.DataFrame) -> Optional[float]:
    """Operating CF - CapEx = Free Cash Flow (most recent year)."""
    try:
        if cf is None or cf.empty:
            return None
        ocf  = _safe_get(cf, "Operating Cash Flow", None)
        capex = _safe_get(cf, "Capital Expenditure", 0)
        if ocf is None:
            return None
        return float(ocf) - abs(float(capex))
    except Exception:
        return None


def _safe_get(df: pd.DataFrame, key: str, default):
    """Safely get first value from a DataFrame row."""
    try:
        if df is None or df.empty or key not in df.index:
            return default
        val = df.loc[key].dropna()
        return float(val.iloc[0]) if not val.empty else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Earnings data (stocks only)
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_data(symbol: str) -> dict:
    """Return EPS history and surprise data."""
    if is_crypto(symbol):
        return {}
    try:
        ticker = yf.Ticker(symbol)
        # earnings_dates replaces the deprecated quarterly_earnings
        # Columns: Reported EPS, EPS Estimate, Surprise(%)
        dates = ticker.earnings_dates
        if dates is None or dates.empty:
            return {}

        # Drop future quarters (no reported EPS yet) and sort oldest→newest
        reported = dates.dropna(subset=["Reported EPS"]).sort_index(ascending=True)
        if reported.empty:
            return {}

        actuals   = reported["Reported EPS"].values
        estimates = reported["EPS Estimate"].values

        beats = [
            float(a) > float(e)
            for a, e in zip(actuals, estimates)
            if pd.notna(a) and pd.notna(e) and float(e) != 0
        ]
        beat_rate  = sum(beats) / len(beats) if beats else None
        eps_growth = None
        if len(actuals) >= 2:
            first, last = float(actuals[0]), float(actuals[-1])
            if first != 0:
                eps_growth = (last - first) / abs(first)

        last_estimate = (float(estimates[-1])
                         if len(estimates) and pd.notna(estimates[-1]) else None)

        return {
            "beat_rate":    round(beat_rate, 3) if beat_rate is not None else None,
            "quarters":     len(beats),
            "eps_growth":   round(eps_growth, 4) if eps_growth is not None else None,
            "last_actual":  float(actuals[-1]) if len(actuals) else None,
            "last_estimate": last_estimate,
        }
    except Exception as e:
        logger.error(f"Earnings error for {symbol}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Insider & institutional data
# ─────────────────────────────────────────────────────────────────────────────

def get_insider_data(symbol: str) -> dict:
    """Net insider buying sentiment from recent filings."""
    if is_crypto(symbol):
        return {}
    try:
        ticker   = yf.Ticker(symbol)
        insiders = ticker.insider_transactions
        if insiders is None or insiders.empty:
            return {"net_insider_sentiment": "unknown"}

        recent = insiders.head(20)
        bought = recent[recent["Transaction"].str.contains("Purchase", na=False)]["Value"].sum()
        sold   = recent[recent["Transaction"].str.contains("Sale", na=False)]["Value"].sum()

        net = bought - sold
        sentiment = "bullish" if net > 0 else ("bearish" if net < 0 else "neutral")
        return {
            "net_insider_value": float(net),
            "net_insider_sentiment": sentiment,
            "insider_buys_value": float(bought),
            "insider_sells_value": float(sold),
        }
    except Exception as e:
        logger.error(f"Insider error for {symbol}: {e}")
        return {"net_insider_sentiment": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# Macro indicators
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_data() -> dict:
    """Fetch VIX, 10yr Treasury, DXY, S&P 500 trend."""
    results = {}
    for name, ticker_sym in config.MACRO_TICKERS.items():
        try:
            ticker = yf.Ticker(ticker_sym)
            hist   = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                continue
            close = hist["Close"]
            results[name] = {
                "current":    float(close.iloc[-1]),
                "1m_change":  float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
                              if len(close) >= 21 else None,
                "3m_change":  float((close.iloc[-1] / close.iloc[0] - 1) * 100),
                "above_50ma": bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1])
                              if len(close) >= 50 else None,
            }
        except Exception as e:
            logger.error(f"Macro data error ({name}): {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sector peers
# ─────────────────────────────────────────────────────────────────────────────

# Hand-curated sector peers for competitive comparison
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":            ["AAPL","MSFT","NVDA","AMD","INTC","AVGO","CSCO","ORCL",
                              "TXN","QCOM","NOW","IBM","MU","AMAT","ADBE","CRM","ADSK"],
    "Communication Services":["GOOGL","META","NFLX","DIS","T","VZ","CMCSA"],
    "Consumer Cyclical":     ["AMZN","TSLA","HD","MCD","SBUX","NKE","TGT","LOW"],
    "Consumer Defensive":    ["PG","WMT","COST","KO","PEP","MDLZ","CL","GIS"],
    "Healthcare":            ["JNJ","UNH","PFE","ABBV","MRK","TMO","DHR","LLY",
                              "CVS","AMGN","GILD","BMY"],
    "Financial Services":    ["JPM","BAC","WFC","MS","GS","BLK","V","MA","AXP"],
    "Energy":                ["XOM","CVX","COP","SLB","PSX","EOG","MPC","VLO"],
    "Utilities":             ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL"],
    "Industrials":           ["UPS","FDX","CAT","DE","HON","MMM","GE","LMT","RTX","BA"],
    "Real Estate":           ["AMT","PLD","CCI","EQIX","SPG","O","VICI"],
    "Basic Materials":       ["LIN","APD","SHW","FCX","NEM","DOW","DD"],
}


def get_sector_peers(symbol: str, max_peers: int = 8) -> list[str]:
    """Return peer list for a stock's sector."""
    if is_crypto(symbol):
        return [s for s in config.CRYPTO_UNIVERSE if s != symbol][:max_peers]
    try:
        info   = yf.Ticker(symbol).info or {}
        sector = info.get("sector", "")
        peers  = SECTOR_PEERS.get(sector, [])
        return [p for p in peers if p != symbol][:max_peers]
    except Exception:
        return []


def infer_sector(symbol: str) -> str:
    """Infer sector from the hardcoded SECTOR_PEERS map (no API call)."""
    if is_crypto(symbol):
        return "crypto"
    for sector, peers in SECTOR_PEERS.items():
        if symbol in peers:
            return sector
    return "Unknown"


# Cache: symbol → (median_pe, expiry_timestamp)
_sector_pe_cache: dict[str, tuple[Optional[float], float]] = {}
_SECTOR_PE_TTL = 86400.0  # 24 hours


def get_sector_median_pe(symbol: str) -> Optional[float]:
    """Compute median P/E for sector peers. Results cached for 24 hours."""
    now = time.time()
    cached = _sector_pe_cache.get(symbol)
    if cached is not None and now < cached[1]:
        return cached[0]

    peers = get_sector_peers(symbol)
    pes   = []
    for peer in peers[:6]:   # limit API calls
        try:
            info = yf.Ticker(peer).info or {}
            pe   = info.get("trailingPE")
            if pe and 0 < pe < 200:
                pes.append(pe)
        except Exception:
            pass
        time.sleep(0.1)
    result = float(np.median(pes)) if pes else None
    _sector_pe_cache[symbol] = (result, now + _SECTOR_PE_TTL)
    return result
