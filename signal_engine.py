"""
signal_engine.py — Orchestrates all 10 techniques and aggregates signals.

Flow for each symbol:
  1. Fetch all required market data once (shared across techniques)
  2. Run all 10 techniques concurrently
  3. Count BUY / SELL signals
  4. If either count >= MIN_SIGNALS_TO_TRADE → raise a TradingSignal
  5. Save all individual signals to database
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import config
import database as db
from data.market_data import (
    get_price_history, get_fundamentals, get_earnings_data,
    get_insider_data, get_macro_data, get_sector_peers,
    get_sector_median_pe, is_crypto, infer_sector
)
from techniques import ALL_TECHNIQUES
from techniques.base_technique import TechniqueResult, SIGNAL_BUY, SIGNAL_SELL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data bundle
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_data_bundle(symbol: str) -> dict:
    """Fetch all data once and share across techniques."""
    bundle: dict = {"symbol": symbol}

    # Price history (1 year, daily)
    bundle["price_history"] = get_price_history(symbol, period="1y")

    # Fundamentals & earnings (stocks only)
    if not is_crypto(symbol):
        bundle["fundamentals"]   = get_fundamentals(symbol)
        bundle["earnings"]       = get_earnings_data(symbol)
        bundle["insider"]        = get_insider_data(symbol)
        bundle["sector_peers"]   = get_sector_peers(symbol)
        bundle["sector_median_pe"] = get_sector_median_pe(symbol)
    else:
        bundle["fundamentals"]   = {}
        bundle["earnings"]       = {}
        bundle["insider"]        = {}
        bundle["sector_peers"]   = []
        bundle["sector_median_pe"] = None

    # Macro data (shared across all symbols — cache it)
    bundle["macro"] = _get_cached_macro()

    # Open positions (for portfolio technique) — annotate with sector
    positions = db.get_positions(config.TRADING_MODE)
    for pos in positions:
        pos["sector"] = infer_sector(pos["symbol"])
    bundle["open_positions"] = positions

    return bundle


# Simple in-memory macro cache (refresh every 30 min)
_macro_cache: dict = {}
_macro_last_fetch: float = 0.0
MACRO_CACHE_SECONDS = 1800


def _get_cached_macro() -> dict:
    global _macro_cache, _macro_last_fetch
    now = time.time()
    if not _macro_cache or (now - _macro_last_fetch) > MACRO_CACHE_SECONDS:
        logger.info("Refreshing macro data cache…")
        _macro_cache      = get_macro_data()
        _macro_last_fetch = now
    return _macro_cache


# ─────────────────────────────────────────────────────────────────────────────
# Trading signal output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingSignal:
    symbol:       str
    asset_type:   str            # 'stock' | 'crypto'
    direction:    str            # 'BUY' | 'SELL'
    score:        float          # weighted aggregate score
    confidence:   float
    buy_count:    int
    sell_count:   int
    technique_results: list[TechniqueResult]
    techniques_summary: dict     # for LLM context

    def __str__(self):
        return (f"TradingSignal({self.symbol} {self.direction} | "
                f"score={self.score:.1f} buy={self.buy_count}/10 "
                f"sell={self.sell_count}/10 conf={self.confidence:.0%})")


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:

    def __init__(self):
        self.techniques = [T() for T in ALL_TECHNIQUES]

    def analyse_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """
        Run all 10 techniques on a symbol and return a TradingSignal
        if MIN_SIGNALS_TO_TRADE or more agree on direction.
        Returns None if no strong consensus.
        """
        asset_type = "crypto" if is_crypto(symbol) else "stock"
        logger.info(f"Analysing {symbol} ({asset_type})…")

        # ── Fetch all data ────────────────────────────────────────────────────
        try:
            data = _fetch_data_bundle(symbol)
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            return None

        # ── Run techniques concurrently ───────────────────────────────────────
        results: list[TechniqueResult] = []

        def run_technique(technique):
            try:
                return technique.analyse(symbol, data)
            except Exception as e:
                logger.warning(f"{technique.name} error on {symbol}: {e}")
                from techniques.base_technique import TechniqueResult
                return TechniqueResult(
                    name=technique.name, signal="NEUTRAL",
                    score=50.0, confidence=0.0,
                    reasoning={"error": str(e)}, applicable=False
                )

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(run_technique, t): t for t in self.techniques}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # ── Persist individual signals ────────────────────────────────────────
        for r in results:
            try:
                db.save_signal(symbol, asset_type, r.name,
                               r.signal, r.score, r.confidence, r.reasoning)
            except Exception as e:
                logger.warning(f"DB signal save error: {e}")

        # ── Vote count ────────────────────────────────────────────────────────
        # Only count votes with meaningful confidence — low-confidence results
        # (e.g. generic crypto fallbacks at 0.20) count as NEUTRAL, not BUY/SELL.
        MIN_VOTE_CONFIDENCE = 0.30
        applicable = [r for r in results if r.applicable]
        buy_votes  = [r for r in applicable if r.is_buy  and r.confidence >= MIN_VOTE_CONFIDENCE]
        sell_votes = [r for r in applicable if r.is_sell and r.confidence >= MIN_VOTE_CONFIDENCE]

        logger.info(f"{symbol}: {len(buy_votes)} BUY / {len(sell_votes)} SELL "
                    f"/ {len(applicable) - len(buy_votes) - len(sell_votes)} NEUTRAL "
                    f"(from {len(applicable)} applicable techniques)")

        # ── Weighted aggregate score ──────────────────────────────────────────
        weighted_score  = 0.0
        total_weight    = 0.0
        for r in applicable:
            w             = config.TECHNIQUE_WEIGHTS.get(r.name, 1.0)
            weighted_score += r.score * w
            total_weight   += w
        avg_score = weighted_score / total_weight if total_weight > 0 else 50.0

        # ── Confidence (average of top contributors) ──────────────────────────
        if buy_votes or sell_votes:
            top = buy_votes if len(buy_votes) >= len(sell_votes) else sell_votes
            confidence = sum(r.confidence for r in top) / len(top)
        else:
            confidence = 0.0

        # ── Direction ─────────────────────────────────────────────────────────
        direction = None
        if len(buy_votes) >= config.MIN_SIGNALS_TO_TRADE:
            direction = SIGNAL_BUY
        elif len(sell_votes) >= config.MIN_SIGNALS_TO_TRADE:
            direction = SIGNAL_SELL

        if direction is None:
            logger.info(f"{symbol}: No consensus — skipping")
            return None

        # ── Build summary dict for LLM judge ──────────────────────────────────
        techniques_summary = {}
        for r in results:
            techniques_summary[r.name] = {
                "signal":     r.signal,
                "score":      r.score,
                "confidence": r.confidence,
                "verdict":    r.reasoning.get("verdict", ""),
                "applicable": r.applicable,
            }

        signal = TradingSignal(
            symbol            = symbol,
            asset_type        = asset_type,
            direction         = direction,
            score             = round(avg_score, 1),
            confidence        = round(confidence, 3),
            buy_count         = len(buy_votes),
            sell_count        = len(sell_votes),
            technique_results = results,
            techniques_summary= techniques_summary,
        )

        logger.info(f"Signal generated: {signal}")
        return signal

    def scan_universe(self, max_workers: int = 3) -> list[TradingSignal]:
        """
        Scan all stocks + crypto in the configured universe.
        Symbols are scanned in parallel (max_workers threads) to cut wall-clock time.
        Returns list of TradingSignals with consensus (skips NEUTRAL).
        """
        all_symbols = (
            list(config.STOCK_UNIVERSE) +
            list(config.CRYPTO_UNIVERSE)
        )

        scan_id = db.start_scan_log()
        errors: list[str] = []
        signals: list[TradingSignal] = []
        total = len(all_symbols)

        logger.info(f"Starting full universe scan: {total} symbols (workers={max_workers})…")

        completed = 0

        def _scan_one(symbol: str):
            nonlocal completed
            time.sleep(0.5)   # stagger to avoid yfinance rate limits (429 errors)
            result = self.analyse_symbol(symbol)
            completed += 1
            logger.info(f"[{completed}/{total}] {symbol} done")
            return symbol, result, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scan_one, sym): sym for sym in all_symbols}
            for future in as_completed(futures):
                try:
                    symbol, signal, _ = future.result()
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    sym = futures[future]
                    error_msg = f"{sym}: {e}"
                    errors.append(error_msg)
                    logger.error(f"Scan error — {error_msg}")

        db.finish_scan_log(
            scan_id,
            symbols_scanned=total,
            signals_generated=len(signals),
            trades_executed=0,   # will be updated by trading engine
            errors="; ".join(errors) if errors else "",
        )

        logger.info(f"Scan complete: {len(signals)} signals from {total} symbols")
        return signals
