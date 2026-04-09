"""
trading_engine.py — Master orchestrator.

Connects: SignalEngine → LLMJudge → RiskManager → PaperTrader / LiveBrokers

Runs on a schedule (every SCAN_INTERVAL minutes).
Handles both stock (Alpaca) and crypto (Binance) assets.
"""

import logging
import time
from datetime import datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import config
import database as db
from signal_engine import SignalEngine, TradingSignal
from risk_manager import RiskManager
from paper_trader import PaperTrader
from data.market_data import get_current_price

US_EASTERN = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Unified trading engine for paper and live modes.
    """

    def __init__(self, mode: str = None):
        self.mode          = mode or config.TRADING_MODE
        self.signal_engine = SignalEngine()
        self.risk_manager  = RiskManager()

        # Paper mode
        self.paper_trader  = PaperTrader() if self.mode == "paper" else None

        # Live broker (Alpaca — stocks only)
        self._alpaca = None
        if self.mode == "live":
            self._init_live_brokers()

        self.risk_manager.set_brokers(self._alpaca, None)
        logger.info(f"TradingEngine ready | mode={self.mode}")

    def _init_live_brokers(self):
        from broker.alpaca_broker import AlpacaBroker
        self._alpaca = AlpacaBroker()

    # ── Main scan & trade loop ────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """
        Execute one complete scan → signal → judge → trade cycle.
        Returns a summary dict.
        """
        start_time = datetime.utcnow()
        logger.info(f"=== CYCLE START [{start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] ===")

        summary = {
            "started_at":     start_time.isoformat(),
            "signals_found":  0,
            "trades_executed": 0,
            "trades_skipped":  0,
            "positions_closed": 0,
            "errors":          [],
        }

        # ── Step 1: Monitor existing positions (stop-loss / take-profit) ────────
        if self.mode == "paper" and self.paper_trader:
            closed = self.paper_trader.monitor_positions(self.risk_manager)
            summary["positions_closed"] = len(closed)
            for c in closed:
                pnl = c.get("pnl", 0)
                logger.info(f"Position closed: {c['symbol']} via {c['reason']} "
                            f"P&L={'+'if pnl>=0 else ''}{pnl:.2f}")

        elif self.mode == "live":
            self._monitor_live_positions()

        # ── Step 1b: End-of-day — close profitable positions when market closed
        if self._is_eod_window():
            eod_closed = self._close_profitable_eod()
            summary["positions_closed"] += len(eod_closed)
            if eod_closed:
                logger.info(f"[EOD] Closed {len(eod_closed)} profitable position(s) "
                            f"before market close")
            logger.info("Market is closed — skipping scan and new trades")
            summary["finished_at"] = datetime.utcnow().isoformat()
            return summary

        # ── Step 2: Scan universe for signals ─────────────────────────────────
        try:
            signals = self.signal_engine.scan_universe()
            summary["signals_found"] = len(signals)
        except Exception as e:
            logger.error(f"Signal scan failed: {e}")
            summary["errors"].append(str(e))
            return summary

        # ── Step 2b: High-conviction signal exits (reuses scan results) ───────
        signal_exits = self._check_signal_exits(signals)
        summary["positions_closed"] += len(signal_exits)

        # ── Step 3: Process each signal (best quality first) ─────────────────
        signals = sorted(signals, key=lambda s: s.score * s.confidence, reverse=True)
        for signal in signals:
            try:
                executed = self._process_signal(signal)
                if executed:
                    summary["trades_executed"] += 1
                else:
                    summary["trades_skipped"] += 1
            except Exception as e:
                logger.error(f"Signal processing error ({signal.symbol}): {e}")
                summary["errors"].append(f"{signal.symbol}: {str(e)}")

        summary["finished_at"] = datetime.utcnow().isoformat()
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"=== CYCLE DONE | {elapsed:.0f}s | "
            f"signals={summary['signals_found']} "
            f"executed={summary['trades_executed']} "
            f"skipped={summary['trades_skipped']} ==="
        )
        return summary

    # ── Signal processing ─────────────────────────────────────────────────────

    def _process_signal(self, signal: TradingSignal) -> bool:
        """
        Process a single trading signal end-to-end.
        Returns True if a trade was executed.
        """
        symbol = signal.symbol
        logger.info(f"Processing signal: {signal}")

        # Only handle BUY signals for now (no shorting)
        if signal.direction != "BUY":
            logger.info(f"{symbol}: SELL signal — no short positions in this version, skip")
            return False

        # Quality filters — require high conviction before trading
        if signal.score < 65:
            logger.info(f"{symbol}: Score {signal.score:.1f} below 65 threshold — skip")
            return False
        if signal.confidence < 0.55:
            logger.info(f"{symbol}: Confidence {signal.confidence:.0%} below 55% threshold — skip")
            return False

        # Get current price
        price = get_current_price(symbol)
        if not price:
            logger.warning(f"{symbol}: Cannot get current price — skip")
            return False

        # Get price history for ATR calculation
        from data.market_data import get_price_history
        hist = get_price_history(symbol, period="3mo")

        # Calculate trade parameters
        params = self.risk_manager.calculate_trade(
            symbol        = symbol,
            direction     = signal.direction,
            price_history = hist,
            current_price = price,
        )

        if not params:
            logger.info(f"{symbol}: Risk checks failed — skip")
            return False

        # Execute the trade
        if self.mode == "paper":
            return self._execute_paper(params, signal)
        else:
            return self._execute_live(params, signal)

    # ── High-conviction signal exits ──────────────────────────────────────────

    SIGNAL_EXIT_THRESHOLD = 6   # out of 10 techniques must agree SELL

    def _check_signal_exits(self, scan_signals: list) -> list[dict]:
        """
        Check open positions against already-computed scan signals.
        Reuses results from scan_universe — no extra analysis needed.

        Close only if ALL of:
          1. Symbol appeared in the scan with a SELL direction
          2. sell_count >= 6 (high conviction, not noise)
          3. Current price > entry price (position profitable — let stop handle losers)
        """
        positions = db.get_positions(self.mode)
        if not positions:
            return []

        # Build lookup from scan results (symbols with consensus SELL are already here)
        sell_signals = {s.symbol: s for s in scan_signals if s.direction == "SELL"}

        closed = []
        for pos in positions:
            symbol = pos["symbol"]
            try:
                signal = sell_signals.get(symbol)
                if signal is None or signal.sell_count < self.SIGNAL_EXIT_THRESHOLD:
                    continue

                current_price = get_current_price(symbol)
                if not current_price:
                    continue

                entry_price = pos["entry_price"]
                if current_price <= entry_price:
                    logger.info(
                        f"[SIGNAL EXIT] {symbol}: {signal.sell_count}/10 SELL but position "
                        f"at a loss (entry=${entry_price:.4f} current=${current_price:.4f}) "
                        f"— letting stop handle it"
                    )
                    continue

                pnl = (current_price - entry_price) * pos["quantity"]
                logger.info(
                    f"[SIGNAL EXIT] {symbol}: {signal.sell_count}/10 techniques SELL + "
                    f"profitable (P&L=+${pnl:.2f}) — closing early"
                )

                if self.mode == "paper":
                    realised_pnl = self.paper_trader.execute_sell(symbol, reason="signal_reversal")
                    closed.append({"symbol": symbol, "reason": "signal_reversal",
                                   "pnl": realised_pnl, "sell_count": signal.sell_count})
                else:
                    self._close_live_position(pos, current_price, reason="signal_reversal")
                    closed.append({"symbol": symbol, "reason": "signal_reversal",
                                   "pnl": pnl, "sell_count": signal.sell_count})

            except Exception as e:
                logger.error(f"Signal exit check error ({symbol}): {e}")

        return closed

    # ── Execution: Paper ──────────────────────────────────────────────────────

    def _execute_paper(self, params, signal: TradingSignal) -> bool:
        trade_id = self.paper_trader.execute_buy(
            params             = params,
            llm_reasoning      = "",
            techniques_summary = signal.techniques_summary,
        )
        if trade_id:
            logger.info(f"[PAPER] Trade #{trade_id} opened: "
                        f"{params.symbol} BUY {params.quantity} @ ${params.entry_price:.4f}")
            return True
        return False

    # ── Execution: Live ───────────────────────────────────────────────────────

    def _execute_live(self, params, signal: TradingSignal) -> bool:
        symbol = params.symbol

        if self._alpaca is None:
            logger.error(f"No broker available for {symbol}")
            return False

        order_id = self._alpaca.buy(
            symbol      = symbol,
            quantity    = params.quantity,
            stop_loss   = params.stop_loss,
            take_profit = params.take_profit,
        )

        if not order_id:
            logger.error(f"[LIVE] Order failed for {symbol}")
            return False

        trade_id = db.open_trade(
            symbol             = symbol,
            asset_type         = "stock",
            side               = "BUY",
            quantity           = params.quantity,
            entry_price        = params.entry_price,
            stop_loss          = params.stop_loss,
            take_profit        = params.take_profit,
            mode               = "live",
            llm_reasoning      = "",
            techniques_summary = signal.techniques_summary,
        )

        db.upsert_position(
            symbol        = symbol,
            asset_type    = "stock",
            quantity      = params.quantity,
            entry_price   = params.entry_price,
            current_price = params.entry_price,
            stop_loss     = params.stop_loss,
            take_profit   = params.take_profit,
            trade_id      = trade_id,
            mode          = "live",
        )

        logger.info(f"[LIVE] Trade #{trade_id} executed: "
                    f"{symbol} BUY {params.quantity} @ ${params.entry_price:.4f} | "
                    f"order_id={order_id}")
        return True

    # ── Live close helpers ────────────────────────────────────────────────────

    def _close_live_position(self, pos: dict, current_price: float, reason: str) -> None:
        symbol = pos["symbol"]
        if self._alpaca:
            self._alpaca.sell(symbol, pos["quantity"])

        pnl = (current_price - pos["entry_price"]) * pos["quantity"]
        open_trades = db.get_open_trades("live")
        trade = next((t for t in open_trades if t["symbol"] == symbol), None)
        if trade:
            db.close_trade(trade["id"], current_price, pnl)
        db.remove_position(symbol)
        logger.info(f"[LIVE] Closed {symbol} ({reason}) @ ${current_price:.4f} P&L=${pnl:+.2f}")

    def _partial_close_live_position(self, pos: dict, sell_fraction: float,
                                      current_price: float, reason: str) -> Optional[dict]:
        """
        Sell `sell_fraction` of a live position, keep the rest running.
        After partial close: stop → breakeven, TP → current + 2× stop-distance.
        Returns summary dict or None on failure.
        """
        symbol      = pos["symbol"]
        total_qty   = pos["quantity"]
        entry_price = pos["entry_price"]

        sell_qty      = max(1, int(total_qty * sell_fraction))

        remaining_qty = total_qty - sell_qty
        if remaining_qty <= 0:
            # Round-down leaves nothing — do a full close
            self._close_live_position(pos, current_price, reason=reason)
            return {"pnl": (current_price - entry_price) * total_qty,
                    "full_close": True, "sell_qty": total_qty, "remaining_qty": 0}

        # Execute partial sell via broker
        if self._alpaca:
            self._alpaca.sell(symbol, sell_qty)

        pnl = (current_price - entry_price) * sell_qty

        # Record sold portion as a closed trade
        trade_id = db.open_trade(
            symbol             = symbol,
            asset_type         = "stock",
            side               = "BUY",
            quantity           = sell_qty,
            entry_price        = entry_price,
            stop_loss          = pos["stop_loss"],
            take_profit        = pos["take_profit"],
            mode               = "live",
            llm_reasoning      = f"Partial close ({sell_fraction:.0%}): {reason}",
            techniques_summary = {},
        )
        db.close_trade(trade_id, current_price, pnl)

        # New risk levels for remaining position
        stop_distance = entry_price - pos["stop_loss"]
        new_stop      = entry_price                        # breakeven
        new_tp        = current_price + stop_distance * 2  # fresh 2:1

        db.update_position_after_partial(symbol, remaining_qty, new_stop, new_tp)

        # Place new stop/TP orders for the remaining quantity via broker
        if broker:
            try:
                broker.sell(symbol, 0)   # cancel existing orders (no-op if not supported)
            except Exception:
                pass
            try:
                broker.buy(symbol, 0,
                           stop_loss=new_stop,
                           take_profit=new_tp)  # update bracket — broker-specific
            except Exception:
                pass

        logger.info(
            f"[LIVE] PARTIAL SELL {symbol} ({sell_fraction:.0%}) | "
            f"sold {sell_qty} @ ${current_price:.4f} | P&L=+${pnl:.2f} | "
            f"remaining={remaining_qty} | new SL=${new_stop:.4f} (breakeven) | "
            f"new TP=${new_tp:.4f}"
        )
        return {"pnl": pnl, "full_close": False,
                "sell_qty": sell_qty, "remaining_qty": remaining_qty,
                "new_stop": new_stop, "new_tp": new_tp}

    # ── End-of-day close ─────────────────────────────────────────────────────

    def _is_eod_window(self) -> bool:
        """True if market is closed on a weekday (after 4 PM ET or before 9:30 AM ET).
        Closes profitable positions on the first scan after market close."""
        now_et = datetime.now(US_EASTERN)
        if now_et.weekday() >= 5:   # weekend — close any remaining profitable positions
            return True
        market_open  = dtime(9, 30)
        market_close = dtime(16, 0)
        # After market close or before market open
        return now_et.time() >= market_close or now_et.time() < market_open

    def _close_profitable_eod(self) -> list[dict]:
        """
        Close all profitable open positions before market close.
        Losers are left alone — stops handle them.
        Re-entry next morning happens automatically if signal is still strong.
        """
        positions = db.get_positions(self.mode)
        closed = []

        for pos in positions:
            symbol      = pos["symbol"]
            entry_price = pos["entry_price"]

            current_price = get_current_price(symbol)
            if not current_price:
                continue

            if current_price < entry_price:
                logger.info(f"[EOD] {symbol}: at a loss (entry=${entry_price:.2f} "
                            f"current=${current_price:.2f}) — leaving stop to handle it")
                continue

            pnl = (current_price - entry_price) * pos["quantity"]
            logger.info(f"[EOD] Closing profitable {symbol} @ ${current_price:.2f} "
                        f"P&L=+${pnl:.2f} (market closes soon)")

            if self.mode == "paper" and self.paper_trader:
                realised = self.paper_trader.execute_sell(symbol, reason="eod_close")
                closed.append({"symbol": symbol, "reason": "eod_close",
                               "pnl": realised or pnl})
            else:
                self._close_live_position(pos, current_price, reason="eod_close")
                closed.append({"symbol": symbol, "reason": "eod_close", "pnl": pnl})

        return closed

    # ── Live position monitoring ──────────────────────────────────────────────

    def _monitor_live_positions(self):
        """Check and close live positions that hit stop/TP. Updates trailing stops."""
        positions = db.get_positions("live")
        for pos in positions:
            symbol = pos["symbol"]
            price  = get_current_price(symbol)
            if not price:
                continue

            db.update_position_price(symbol, price)

            new_stop = self.risk_manager.trailing_stop_update(pos, price)
            if new_stop is not None:
                db.update_position_stop(symbol, new_stop)
                pos["stop_loss"] = new_stop
                logger.info(f"[LIVE] Trailing stop raised: {symbol} → ${new_stop:.4f}")

            exit_reason = self.risk_manager.check_stop_take_profit(pos, price)

            if exit_reason == "take_profit":
                score, conf = db.get_latest_signal_quality(symbol)
                sell_fraction = 0.30 if (score >= 80 and conf >= 0.80) else 0.50
                logger.info(
                    f"[LIVE] TP hit {symbol} | signal quality score={score:.1f} "
                    f"conf={conf:.0%} → selling {sell_fraction:.0%}"
                )
                self._partial_close_live_position(pos, sell_fraction, price,
                                                  reason="take_profit")
            elif exit_reason == "stop_loss":
                self._close_live_position(pos, price, reason="stop_loss")

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current bot status summary."""
        if self.mode == "paper":
            account = self.paper_trader.get_account()
        else:
            account = {"mode": "live"}

        positions = db.get_positions(self.mode)
        pnl_stats = db.get_pnl_summary(self.mode)

        return {
            "mode":      self.mode,
            "account":   account,
            "positions": positions,
            "pnl_stats": pnl_stats,
        }
