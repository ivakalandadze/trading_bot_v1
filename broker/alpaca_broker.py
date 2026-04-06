"""
broker/alpaca_broker.py — Alpaca Markets integration for US stocks.

Uses the NEW alpaca-py SDK (replaces deprecated alpaca-trade-api).
Install: pip install alpaca-py

Paper trading: set ALPACA_BASE_URL=https://paper-api.alpaca.markets in .env
Live trading:  set ALPACA_BASE_URL=https://api.alpaca.markets in .env
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Wrapper around alpaca-py (the modern Alpaca SDK) for stock trading.
    Auto-selects paper vs live based on ALPACA_BASE_URL in config.
    """

    def __init__(self):
        self._trading_client = None
        self._data_client     = None
        self._is_paper        = "paper" in config.ALPACA_BASE_URL
        self._init_clients()

    def _init_clients(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            logger.warning("Alpaca API keys not configured — stock trading disabled")
            return
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key    = config.ALPACA_API_KEY,
                secret_key = config.ALPACA_SECRET_KEY,
                paper      = self._is_paper,
            )
            self._data_client = StockHistoricalDataClient(
                api_key    = config.ALPACA_API_KEY,
                secret_key = config.ALPACA_SECRET_KEY,
            )

            acct = self._trading_client.get_account()
            mode = "PAPER" if self._is_paper else "LIVE"
            logger.info(f"Alpaca {mode} connected | "
                        f"equity=${float(acct.equity):,.2f} | "
                        f"buying_power=${float(acct.buying_power):,.2f}")

        except ImportError:
            logger.error(
                "alpaca-py not installed. Run: pip install alpaca-py\n"
                "(Note: the old 'alpaca-trade-api' package is deprecated)"
            )
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> Optional[dict]:
        if not self._trading_client:
            return None
        try:
            acct = self._trading_client.get_account()
            return {
                "equity":          float(acct.equity),
                "cash":            float(acct.cash),
                "buying_power":    float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "pnl_today":       float(acct.equity) - float(acct.last_equity),
                "status":          str(acct.status),
            }
        except Exception as e:
            logger.error(f"Alpaca get_account error: {e}")
            return None

    def get_buying_power(self) -> float:
        account = self.get_account()
        return account["buying_power"] if account else 0.0

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy(self, symbol: str, quantity: float,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None) -> Optional[str]:
        """
        Submit a market buy order.
        Uses a bracket order if stop_loss and take_profit are provided.
        Returns order ID string on success, None on failure.
        """
        if not self._trading_client:
            logger.warning(f"Alpaca not available — cannot buy {symbol}")
            return None
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, OrderClass,
                TakeProfitRequest, StopLossRequest
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = int(max(1, quantity))   # whole shares only

            if stop_loss and take_profit:
                req = MarketOrderRequest(
                    symbol        = symbol,
                    qty           = qty,
                    side          = OrderSide.BUY,
                    time_in_force = TimeInForce.GTC,
                    order_class   = OrderClass.BRACKET,
                    take_profit   = TakeProfitRequest(
                        limit_price = round(take_profit, 2)
                    ),
                    stop_loss     = StopLossRequest(
                        stop_price = round(stop_loss, 2)
                    ),
                )
            else:
                req = MarketOrderRequest(
                    symbol        = symbol,
                    qty           = qty,
                    side          = OrderSide.BUY,
                    time_in_force = TimeInForce.DAY,
                )

            order = self._trading_client.submit_order(req)
            logger.info(f"Alpaca BUY submitted: {symbol} x{qty} | id={order.id}")
            return str(order.id)

        except Exception as e:
            logger.error(f"Alpaca buy error ({symbol}): {e}")
            return None

    def sell(self, symbol: str, quantity: float) -> Optional[str]:
        """Submit a market sell order."""
        if not self._trading_client:
            return None
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = int(max(1, quantity))
            req = MarketOrderRequest(
                symbol        = symbol,
                qty           = qty,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.DAY,
            )
            order = self._trading_client.submit_order(req)
            logger.info(f"Alpaca SELL submitted: {symbol} x{qty} | id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Alpaca sell error ({symbol}): {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Close entire position for a symbol."""
        if not self._trading_client:
            return False
        try:
            self._trading_client.close_position(symbol)
            logger.info(f"Alpaca position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Alpaca close_position error ({symbol}): {e}")
            return False

    def close_all_positions(self) -> None:
        if not self._trading_client:
            return
        try:
            self._trading_client.close_all_positions(cancel_orders=True)
            logger.info("All Alpaca positions closed")
        except Exception as e:
            logger.error(f"Alpaca close_all_positions error: {e}")

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        if not self._trading_client:
            return []
        try:
            positions = self._trading_client.get_all_positions()
            return [
                {
                    "symbol":         p.symbol,
                    "qty":            float(p.qty),
                    "avg_entry":      float(p.avg_entry_price),
                    "current_price":  float(p.current_price),
                    "market_value":   float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Alpaca get_positions error: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────────────

    def cancel_all_orders(self) -> None:
        if not self._trading_client:
            return
        try:
            self._trading_client.cancel_orders()
            logger.info("All Alpaca orders cancelled")
        except Exception as e:
            logger.error(f"Alpaca cancel_orders error: {e}")

    def get_order_status(self, order_id: str) -> Optional[str]:
        if not self._trading_client:
            return None
        try:
            import uuid
            order = self._trading_client.get_order_by_id(uuid.UUID(order_id))
            return str(order.status)
        except Exception as e:
            logger.error(f"Alpaca order status error: {e}")
            return None

    # ── Market status ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        if not self._trading_client:
            return False
        try:
            clock = self._trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Alpaca clock error: {e}")
            return False

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest trade price via Alpaca data API."""
        if not self._data_client:
            return None
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            req    = StockLatestTradeRequest(symbol_or_symbols=symbol)
            result = self._data_client.get_stock_latest_trade(req)
            return float(result[symbol].price)
        except Exception as e:
            logger.error(f"Alpaca price error ({symbol}): {e}")
            return None

    def is_available(self) -> bool:
        return self._trading_client is not None
