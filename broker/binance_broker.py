"""
broker/binance_broker.py — Binance exchange integration for crypto.

Supports both Testnet (paper equivalent) and Live trading.
Set BINANCE_TESTNET=True in .env for safe paper trading.
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Minimum order sizes (notional USD) by asset
MIN_NOTIONAL = {
    "BTCUSDT": 10.0,
    "ETHUSDT": 10.0,
    "DEFAULT": 5.0,
}


class BinanceBroker:
    """
    Wrapper around python-binance for crypto trading.
    Testnet URL: https://testnet.binance.vision
    """

    def __init__(self):
        self._client = None
        self._init_client()

    def _init_client(self):
        if not config.BINANCE_API_KEY or not config.BINANCE_SECRET_KEY:
            logger.warning("Binance API keys not configured — crypto trading disabled")
            return
        try:
            from binance.client import Client
            if config.BINANCE_TESTNET:
                self._client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_SECRET_KEY,
                    testnet=True,
                )
                logger.info("Binance TESTNET connected")
            else:
                self._client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_SECRET_KEY,
                )
                logger.info("Binance LIVE connected")
        except ImportError:
            logger.error("python-binance not installed. Run: pip install python-binance")
        except Exception as e:
            logger.error(f"Binance connection error: {e}")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        if not self._client:
            return 0.0
        try:
            balance = self._client.get_asset_balance(asset=asset)
            return float(balance["free"]) if balance else 0.0
        except Exception as e:
            logger.error(f"Binance get_balance error: {e}")
            return 0.0

    def get_all_balances(self) -> dict:
        if not self._client:
            return {}
        try:
            account = self._client.get_account()
            return {
                b["asset"]: float(b["free"])
                for b in account["balances"]
                if float(b["free"]) > 0 or float(b["locked"]) > 0
            }
        except Exception as e:
            logger.error(f"Binance get_all_balances error: {e}")
            return {}

    # ── Current price ─────────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> Optional[float]:
        sym = symbol.replace("/", "").upper()
        if not self._client:
            # Fallback: public REST
            try:
                import requests
                r = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={sym}",
                    timeout=5
                )
                return float(r.json()["price"])
            except Exception:
                return None
        try:
            ticker = self._client.get_symbol_ticker(symbol=sym)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Binance price error ({symbol}): {e}")
            return None

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy(self, symbol: str, quantity: float,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None) -> Optional[str]:
        """
        Market buy on Binance.
        If both stop_loss and take_profit are set, places an OCO sell order.
        If only stop_loss is set, places a stop-market sell.
        Returns order ID string on success.
        """
        if not self._client:
            logger.warning(f"Binance not available — cannot buy {symbol}")
            return None

        sym = symbol.replace("/", "").upper()
        qty = self._round_quantity(sym, quantity)
        try:
            order = self._client.order_market_buy(
                symbol   = sym,
                quantity = qty,
            )
            order_id = str(order["orderId"])
            logger.info(f"Binance BUY: {sym} x{quantity:.6f} | id={order_id}")

            if stop_loss and take_profit:
                self._place_oco_sell(sym, qty, stop_loss, take_profit)
            elif stop_loss:
                self._place_stop_loss(sym, qty, stop_loss)

            return order_id

        except Exception as e:
            logger.error(f"Binance buy error ({symbol}): {e}")
            return None

    def sell(self, symbol: str, quantity: float) -> Optional[str]:
        """Market sell to close a long position."""
        if not self._client:
            return None
        sym = symbol.replace("/", "").upper()
        try:
            # Get actual held quantity to avoid over-selling
            asset       = sym.replace("USDT", "").replace("BTC", "")
            held        = self.get_balance(asset)
            sell_qty    = min(quantity, held)
            sell_qty    = self._round_quantity(sym, sell_qty)

            if sell_qty <= 0:
                logger.warning(f"No {asset} balance to sell")
                return None

            order = self._client.order_market_sell(
                symbol   = sym,
                quantity = sell_qty,
            )
            order_id = str(order["orderId"])
            logger.info(f"Binance SELL: {sym} x{sell_qty:.6f} | id={order_id}")
            return order_id

        except Exception as e:
            logger.error(f"Binance sell error ({symbol}): {e}")
            return None

    def _place_oco_sell(self, symbol: str, quantity: float,
                        stop_price: float, take_profit: float):
        """Place an OCO order: take-profit limit sell + stop-loss.

        stopLimitPrice is set 0.5% below stopPrice so the limit still fills
        if price gaps slightly through the stop trigger.
        """
        qty = self._round_quantity(symbol, quantity)
        tp_price       = self._round_price(symbol, take_profit)
        sl_trigger     = self._round_price(symbol, stop_price)
        sl_limit       = self._round_price(symbol, stop_price * 0.995)
        try:
            self._client.create_oco_order(
                symbol        = symbol,
                side          = "SELL",
                quantity      = qty,
                price         = tp_price,
                stopPrice     = sl_trigger,
                stopLimitPrice= sl_limit,
                stopLimitTimeInForce = "GTC",
            )
            logger.info(f"Binance OCO sell set: {symbol} TP={tp_price} SL={sl_price}")
        except Exception as e:
            logger.warning(f"Binance OCO order failed ({symbol}): {e} — falling back to stop-loss only")
            self._place_stop_loss(symbol, quantity, stop_price)

    def _place_stop_loss(self, symbol: str, quantity: float, stop_price: float):
        """Place a stop-loss limit order."""
        qty = self._round_quantity(symbol, quantity)
        price_str = self._round_price(symbol, stop_price)
        try:
            self._client.create_order(
                symbol        = symbol,
                side          = "SELL",
                type          = "STOP_LOSS_LIMIT",
                quantity      = qty,
                price         = price_str,
                stopPrice     = price_str,
                timeInForce   = "GTC",
            )
            logger.info(f"Binance stop-loss set at {stop_price} for {symbol}")
        except Exception as e:
            logger.warning(f"Binance stop-loss order failed ({symbol}): {e}")

    def cancel_all_open_orders(self, symbol: str) -> None:
        if not self._client:
            return
        sym = symbol.replace("/", "").upper()
        try:
            self._client.cancel_open_orders(symbol=sym)
        except Exception as e:
            logger.error(f"Binance cancel orders error ({symbol}): {e}")

    # ── Symbol info helpers ───────────────────────────────────────────────────

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to exchange step size."""
        try:
            info      = self._client.get_symbol_info(symbol)
            lot_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
            step      = float(lot_filter["stepSize"])
            if step > 0:
                precision = len(str(step).rstrip("0").split(".")[-1])
                return round(quantity - (quantity % step), precision)
        except Exception:
            pass
        return round(quantity, 6)

    def _round_price(self, symbol: str, price: float) -> str:
        """Round price to exchange tick size."""
        try:
            info        = self._client.get_symbol_info(symbol)
            price_filter = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
            tick        = float(price_filter["tickSize"])
            if tick > 0:
                precision = len(str(tick).rstrip("0").split(".")[-1])
                rounded   = round(price - (price % tick), precision)
                return str(rounded)
        except Exception:
            pass
        return str(round(price, 2))

    # ── Order status ──────────────────────────────────────────────────────────

    def get_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        if not self._client:
            return None
        sym = symbol.replace("/", "").upper()
        try:
            order = self._client.get_order(symbol=sym, orderId=int(order_id))
            return order["status"]
        except Exception as e:
            logger.error(f"Binance order status error: {e}")
            return None

    def is_available(self) -> bool:
        return self._client is not None
