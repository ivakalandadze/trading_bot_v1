"""
risk_manager.py — Position sizing and risk controls.

Rules:
  • Risk 2% of total capital per trade
  • Stop-loss based on ATR (Average True Range) × 1.5
  • Take-profit = 2× the dollar risk (2:1 R/R minimum)
  • Never exceed MAX_OPEN_POSITIONS
  • Never risk more than 6% of capital across all open positions
  • Daily loss limit: stop trading if total P&L drops below -5% today
  • Skip stock trades when US market is closed
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime, date
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import database as db
from data.market_data import get_current_price, is_crypto

logger = logging.getLogger(__name__)

MAX_PORTFOLIO_RISK   = 0.06   # max 6% of capital across all open positions
DAILY_LOSS_LIMIT     = -0.05  # halt if daily P&L < -5%
ATR_PERIOD           = 14
ATR_MULTIPLIER       = 1.5    # stop-loss = entry ± ATR * multiplier
MIN_RR_RATIO         = 1.5    # minimum risk/reward to accept a trade

US_EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


@dataclass
class TradeParameters:
    symbol:       str
    side:         str         # 'BUY' | 'SELL'
    quantity:     float
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_amount:  float       # $ amount risked on this trade
    rr_ratio:     float       # risk/reward ratio


class RiskManager:

    def __init__(self, alpaca_broker=None, binance_broker=None):
        self.mode = config.TRADING_MODE
        self._alpaca = alpaca_broker
        self._binance = binance_broker

    def set_brokers(self, alpaca_broker=None, binance_broker=None):
        """Allow TradingEngine to inject live broker references after init."""
        if alpaca_broker is not None:
            self._alpaca = alpaca_broker
        if binance_broker is not None:
            self._binance = binance_broker

    # ── Main entry point ──────────────────────────────────────────────────────

    def calculate_trade(self, symbol: str, direction: str,
                        price_history: pd.DataFrame,
                        current_price: Optional[float] = None) -> Optional[TradeParameters]:
        """
        Calculate position size and levels for a proposed trade.
        Returns None if the trade fails any risk check.
        """
        entry = current_price or get_current_price(symbol)
        if not entry or entry <= 0:
            logger.warning(f"Cannot get price for {symbol}")
            return None

        # ── Pre-trade checks ──────────────────────────────────────────────────
        if not self._pre_trade_checks(symbol):
            return None

        # ── ATR-based stop-loss ───────────────────────────────────────────────
        atr = self._calc_atr(price_history)
        if atr is None or atr <= 0:
            atr = entry * 0.02   # fallback: 2% of price

        stop_distance = atr * ATR_MULTIPLIER

        if direction == "BUY":
            stop_loss   = entry - stop_distance
            take_profit = entry + stop_distance * config.TAKE_PROFIT_RATIO
        else:   # SELL (short)
            stop_loss   = entry + stop_distance
            take_profit = entry - stop_distance * config.TAKE_PROFIT_RATIO

        # ── R/R check ─────────────────────────────────────────────────────────
        rr_ratio = abs(take_profit - entry) / abs(entry - stop_loss)
        if rr_ratio < MIN_RR_RATIO:
            logger.info(f"{symbol}: R/R {rr_ratio:.2f} below minimum {MIN_RR_RATIO} — skip")
            return None

        # ── Position sizing ───────────────────────────────────────────────────
        capital     = self._get_available_capital()
        risk_amount = capital * config.RISK_PER_TRADE
        risk_per_share = abs(entry - stop_loss)

        if risk_per_share <= 0:
            logger.warning(f"{symbol}: Zero risk per share — skip")
            return None

        quantity = risk_amount / risk_per_share

        # For crypto: allow fractional, round to 6 decimals
        # For stocks: round to nearest whole share
        if is_crypto(symbol):
            quantity = round(quantity, 6)
        else:
            quantity = max(1, int(quantity))

        # ── Portfolio risk cap ────────────────────────────────────────────────
        if not self._portfolio_risk_check(risk_amount, capital):
            return None

        return TradeParameters(
            symbol      = symbol,
            side        = direction,
            quantity    = quantity,
            entry_price = round(entry, 6),
            stop_loss   = round(stop_loss, 6),
            take_profit = round(take_profit, 6),
            risk_amount = round(risk_amount, 4),
            rr_ratio    = round(rr_ratio, 2),
        )

    # ── Checks ────────────────────────────────────────────────────────────────

    def _pre_trade_checks(self, symbol: str) -> bool:
        """Gate checks before calculating trade parameters."""
        if not is_crypto(symbol) and not self._is_market_open():
            logger.info(f"{symbol}: US stock market is closed — skip")
            return False

        positions = db.get_positions(self.mode)

        if any(p["symbol"] == symbol for p in positions):
            logger.info(f"{symbol}: Already have an open position — skip")
            return False

        open_trades = db.get_open_trades(self.mode)
        if any(t["symbol"] == symbol for t in open_trades):
            logger.info(f"{symbol}: Open trade already exists — skip duplicate")
            return False

        if len(positions) >= config.MAX_OPEN_POSITIONS:
            logger.info(f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached — skip")
            return False

        if self._daily_loss_exceeded():
            logger.warning("Daily loss limit reached — halting new trades")
            return False

        return True

    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open (weekdays 9:30-16:00 ET)."""
        if self._alpaca is not None:
            try:
                return self._alpaca.is_market_open()
            except Exception:
                pass
        now_et = datetime.now(US_EASTERN)
        if now_et.weekday() >= 5:
            return False
        return MARKET_OPEN <= now_et.time() <= MARKET_CLOSE

    def _portfolio_risk_check(self, new_risk: float, capital: float) -> bool:
        """Ensure total portfolio risk stays below MAX_PORTFOLIO_RISK."""
        positions  = db.get_positions(self.mode)
        # Estimate existing at-risk amounts (rough: risk_per_trade per position)
        existing_risk = len(positions) * capital * config.RISK_PER_TRADE
        total_risk    = existing_risk + new_risk
        max_allowed   = capital * MAX_PORTFOLIO_RISK

        if total_risk > max_allowed:
            logger.info(f"Portfolio risk cap reached "
                        f"({total_risk:.0f} > {max_allowed:.0f}) — skip")
            return False
        return True

    def _daily_loss_exceeded(self) -> bool:
        """Check if today's P&L breaches the daily loss limit."""
        try:
            daily_pnl = db.get_daily_pnl(self.mode, date.today().isoformat())
            capital   = self._get_available_capital()
            daily_ret = daily_pnl / capital if capital > 0 else 0
            return daily_ret < DAILY_LOSS_LIMIT
        except Exception:
            return False

    # ── Capital helpers ───────────────────────────────────────────────────────

    def _get_available_capital(self) -> float:
        """Return available cash from paper account or live broker balance."""
        if self.mode == "paper":
            account = db.get_paper_account()
            return account.get("cash", config.PAPER_CAPITAL)

        if self._alpaca is not None:
            try:
                return self._alpaca.get_buying_power()
            except Exception as e:
                logger.warning(f"Could not get Alpaca buying power: {e}")

        if self._binance is not None:
            try:
                return self._binance.get_balance("USDT")
            except Exception as e:
                logger.warning(f"Could not get Binance balance: {e}")

        logger.warning("No live broker available for capital query — using configured PAPER_CAPITAL as fallback")
        return config.PAPER_CAPITAL

    # ── ATR calculation ───────────────────────────────────────────────────────

    def _calc_atr(self, hist: pd.DataFrame, period: int = ATR_PERIOD) -> Optional[float]:
        """Average True Range over last `period` bars."""
        try:
            if hist is None or len(hist) < period + 1:
                return None
            high  = hist["High"].astype(float)
            low   = hist["Low"].astype(float)
            close = hist["Close"].astype(float)

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low  - close.shift(1)).abs()
            tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr)
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return None

    # ── Stop management (for open positions) ─────────────────────────────────

    def check_stop_take_profit(self, position: dict,
                                current_price: float) -> Optional[str]:
        """
        Check if a position has hit stop-loss or take-profit.
        Returns 'stop_loss', 'take_profit', or None.
        """
        stop = position.get("stop_loss")
        tp   = position.get("take_profit")
        side = "BUY"   # All bot positions are long (no shorting for now)

        if stop and current_price <= stop:
            return "stop_loss"
        if tp and current_price >= tp:
            return "take_profit"
        return None

    def trailing_stop_update(self, position: dict,
                              current_price: float,
                              trail_pct: float = 0.05) -> Optional[float]:
        """
        Update stop-loss with a trailing stop (5% below current high).
        Returns new stop-loss level if it should be raised, else None.
        """
        current_stop = position.get("stop_loss", 0)
        new_stop     = current_price * (1 - trail_pct)
        if new_stop > current_stop:
            return round(new_stop, 6)
        return None
