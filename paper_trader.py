"""
paper_trader.py — Simulated trading engine on real prices.

Maintains a virtual $10,000 account.
Executes trades at real-time prices fetched via yfinance/Binance.
Tracks positions, P&L, stop-losses, and take-profits.
"""

import logging
from datetime import datetime
from typing import Optional

import config
import database as db
from data.market_data import get_current_price, is_crypto
from risk_manager import TradeParameters

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading engine.
    Simulates live execution with real market prices.
    """

    MODE = "paper"

    def __init__(self, starting_capital: float = None):
        self.starting_capital = starting_capital or config.PAPER_CAPITAL
        db.init_paper_account(self.starting_capital)
        logger.info(f"Paper trader initialised | capital=${self.starting_capital:,.2f}")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        account   = db.get_paper_account()
        positions = db.get_positions(self.MODE)

        # Refresh position values
        equity = 0.0
        for pos in positions:
            price = get_current_price(pos["symbol"])
            if price:
                db.update_position_price(pos["symbol"], price)
                equity += price * pos["quantity"]

        db.update_paper_account(account["cash"], equity)
        account = db.get_paper_account()
        account["positions"]  = len(positions)
        account["return_pct"] = ((account["total_value"] - self.starting_capital)
                                  / self.starting_capital * 100)
        return account

    def get_cash(self) -> float:
        return db.get_paper_account().get("cash", self.starting_capital)

    # ── Trade execution ───────────────────────────────────────────────────────

    def execute_buy(self, params: TradeParameters,
                    llm_reasoning: str = "",
                    techniques_summary: dict = None) -> Optional[int]:
        """
        Simulate a market buy.
        Deducts cost from cash, records position.
        Returns trade_id on success.
        """
        cash     = self.get_cash()
        quantity = params.quantity
        cost     = params.entry_price * quantity

        if cost > cash:
            # Adjust quantity to what we can afford (don't mutate the caller's dataclass)
            max_qty  = cash / params.entry_price
            quantity = (round(max_qty * 0.99, 6) if is_crypto(params.symbol)
                        else max(1, int(max_qty * 0.99)))
            cost     = params.entry_price * quantity

        if quantity <= 0 or cost > cash:
            logger.warning(f"Insufficient paper capital (${cash:.2f}) for "
                           f"{params.symbol} (cost ${cost:.2f})")
            return None

        # Record trade
        trade_id = db.open_trade(
            symbol     = params.symbol,
            asset_type = "crypto" if is_crypto(params.symbol) else "stock",
            side       = "BUY",
            quantity   = quantity,
            entry_price = params.entry_price,
            stop_loss  = params.stop_loss,
            take_profit = params.take_profit,
            mode       = self.MODE,
            llm_reasoning  = llm_reasoning,
            techniques_summary = techniques_summary,
        )

        # Update position
        db.upsert_position(
            symbol      = params.symbol,
            asset_type  = "crypto" if is_crypto(params.symbol) else "stock",
            quantity    = quantity,
            entry_price = params.entry_price,
            current_price = params.entry_price,
            stop_loss   = params.stop_loss,
            take_profit = params.take_profit,
            trade_id    = trade_id,
            mode        = self.MODE,
        )

        # Deduct cash
        account = db.get_paper_account()
        db.update_paper_account(
            cash   = account["cash"] - cost,
            equity = account["equity"] + cost,
        )

        logger.info(
            f"[PAPER] BUY  {params.symbol} | "
            f"qty={quantity} @ ${params.entry_price:.4f} | "
            f"cost=${cost:.2f} | SL=${params.stop_loss:.4f} | "
            f"TP=${params.take_profit:.4f} | R/R={params.rr_ratio:.2f}"
        )
        return trade_id

    def execute_partial_sell(self, symbol: str, sell_fraction: float,
                             current_price: float, reason: str = "partial_tp") -> Optional[dict]:
        """
        Sell `sell_fraction` of an open position, keep the rest running.

        After the partial close:
          - Stop-loss moves to entry price (breakeven — free trade)
          - Take-profit reset to current_price + original_stop_distance × 2

        Returns a summary dict, or None on failure.
        """
        positions = db.get_positions(self.MODE)
        pos = next((p for p in positions if p["symbol"] == symbol), None)
        if not pos:
            logger.warning(f"No position to partial-sell: {symbol}")
            return None

        total_qty   = pos["quantity"]
        entry_price = pos["entry_price"]

        sell_qty = total_qty * sell_fraction
        if is_crypto(symbol):
            sell_qty = round(sell_qty, 6)
        else:
            sell_qty = max(1, int(sell_qty))

        remaining_qty = total_qty - sell_qty
        if remaining_qty <= 0:
            # Fraction rounds to nothing — just do a full close
            pnl = self.execute_sell(symbol, reason=reason)
            return {"pnl": pnl, "full_close": True, "sell_qty": total_qty,
                    "remaining_qty": 0}

        proceeds = sell_qty * current_price
        pnl      = (current_price - entry_price) * sell_qty

        # Record the sold portion as a closed trade
        trade_id = db.open_trade(
            symbol             = symbol,
            asset_type         = "crypto" if is_crypto(symbol) else "stock",
            side               = "BUY",
            quantity           = sell_qty,
            entry_price        = entry_price,
            stop_loss          = pos["stop_loss"],
            take_profit        = pos["take_profit"],
            mode               = self.MODE,
            llm_reasoning      = f"Partial close ({sell_fraction:.0%}): {reason}",
            techniques_summary = {},
        )
        db.close_trade(trade_id, current_price, pnl)

        # New levels for remaining position
        stop_distance = entry_price - pos["stop_loss"]
        new_stop      = entry_price                          # breakeven
        new_tp        = current_price + stop_distance * 2   # fresh 2:1 from here

        db.update_position_after_partial(symbol, remaining_qty, new_stop, new_tp)

        # Cash: add proceeds, reduce cost-basis equity
        account = db.get_paper_account()
        db.update_paper_account(
            cash   = account["cash"] + proceeds,
            equity = max(0, account["equity"] - entry_price * sell_qty),
        )

        pnl_str = f"+${pnl:.2f}"
        logger.info(
            f"[PAPER] PARTIAL SELL {symbol} ({sell_fraction:.0%}) | "
            f"sold {sell_qty} @ ${current_price:.4f} | P&L={pnl_str} | "
            f"remaining={remaining_qty} | new SL=${new_stop:.4f} (breakeven) | "
            f"new TP=${new_tp:.4f}"
        )
        return {"pnl": pnl, "full_close": False,
                "sell_qty": sell_qty, "remaining_qty": remaining_qty,
                "new_stop": new_stop, "new_tp": new_tp}

    def execute_sell(self, symbol: str, reason: str = "signal") -> Optional[float]:
        """
        Close an open paper position.
        Returns realised P&L.
        """
        positions = db.get_positions(self.MODE)
        position  = next((p for p in positions if p["symbol"] == symbol), None)

        if not position:
            logger.warning(f"No paper position to sell: {symbol}")
            return None

        # Get current price
        exit_price = get_current_price(symbol)
        if not exit_price:
            logger.error(f"Cannot get exit price for {symbol}")
            return None

        quantity    = position["quantity"]
        entry_price = position["entry_price"]
        proceeds    = exit_price * quantity
        pnl         = (exit_price - entry_price) * quantity

        # Find open trade
        open_trades = db.get_open_trades(self.MODE)
        trade = next(
            (t for t in open_trades if t["symbol"] == symbol), None
        )

        if trade:
            db.close_trade(trade["id"], exit_price, pnl)

        db.remove_position(symbol)

        # Update cash
        account = db.get_paper_account()
        equity_reduction = entry_price * quantity
        db.update_paper_account(
            cash   = account["cash"] + proceeds,
            equity = max(0, account["equity"] - equity_reduction),
        )

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        logger.info(
            f"[PAPER] SELL {symbol} | "
            f"qty={quantity} @ ${exit_price:.4f} | "
            f"P&L={pnl_str} | reason={reason}"
        )
        return pnl

    # ── Position monitoring ───────────────────────────────────────────────────

    def monitor_positions(self, risk_manager) -> list[dict]:
        """
        Check all open positions against stop-loss and take-profit.
        Updates trailing stops for profitable positions.
        Returns list of closed position summaries.
        """
        positions = db.get_positions(self.MODE)
        closed    = []

        for pos in positions:
            symbol = pos["symbol"]
            price  = get_current_price(symbol)

            if not price:
                continue

            db.update_position_price(symbol, price)

            new_stop = risk_manager.trailing_stop_update(pos, price)
            if new_stop is not None:
                db.update_position_stop(symbol, new_stop)
                pos["stop_loss"] = new_stop
                logger.info(f"[PAPER] Trailing stop raised: {symbol} → ${new_stop:.4f}")

            exit_reason = risk_manager.check_stop_take_profit(pos, price)

            if exit_reason == "take_profit":
                # Dynamic partial close: check current signal quality
                score, conf = db.get_latest_signal_quality(symbol)
                sell_fraction = 0.30 if (score >= 80 and conf >= 0.80) else 0.50
                logger.info(
                    f"[PAPER] TP hit {symbol} | signal quality score={score:.1f} "
                    f"conf={conf:.0%} → selling {sell_fraction:.0%}"
                )
                result = self.execute_partial_sell(symbol, sell_fraction, price,
                                                   reason="take_profit")
                if result:
                    closed.append({
                        "symbol": symbol,
                        "reason": f"partial_tp_{sell_fraction:.0%}",
                        "pnl":    result["pnl"],
                        "price":  price,
                    })

            elif exit_reason == "stop_loss":
                pnl = self.execute_sell(symbol, reason="stop_loss")
                closed.append({
                    "symbol": symbol,
                    "reason": "stop_loss",
                    "pnl":    pnl,
                    "price":  price,
                })
                logger.info(f"[PAPER] Position closed ({exit_reason}): "
                            f"{symbol} @ ${price:.4f}")

        return closed

    # ── Performance summary ───────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        account = self.get_account()
        pnl     = db.get_pnl_summary(self.MODE)
        return {
            "account":    account,
            "pnl_stats":  pnl,
            "positions":  db.get_positions(self.MODE),
        }
