"""
T8 — Bain-Style Competitive Advantage Analysis
Compares a company's margins, growth, and returns against sector peers.
Returns a competitive rank and signal. For crypto: volume-rank based.
"""

import logging
import time
import numpy as np
import yfinance as yf
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto, get_sector_peers

logger = logging.getLogger(__name__)

# Peer fundamental metrics cache: symbol → (metrics_dict, expiry_timestamp)
_peer_cache: dict[str, tuple[dict, float]] = {}
_PEER_CACHE_TTL = 86400.0  # 24 hours


def _get_peer_metrics(symbol: str) -> dict:
    """Fetch and cache fundamental metrics for a single peer symbol."""
    now = time.time()
    cached = _peer_cache.get(symbol)
    if cached is not None and now < cached[1]:
        return cached[0]
    try:
        info = yf.Ticker(symbol).info or {}
        metrics = {
            "symbol":           symbol,
            "profit_margin":    info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "revenue_growth":   info.get("revenueGrowth"),
            "operating_margin": info.get("operatingMargins"),
        }
    except Exception:
        metrics = {"symbol": symbol}
    _peer_cache[symbol] = (metrics, now + _PEER_CACHE_TTL)
    return metrics


class BainCompetitive(BaseTechnique):
    name = "T8_Competitive"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        if is_crypto(symbol):
            return self._crypto_competitive(symbol, data)
        return self._stock_competitive(symbol, data)

    # ── Stock competitive analysis ────────────────────────────────────────────

    def _stock_competitive(self, symbol: str, data: dict) -> TechniqueResult:
        fund  = data.get("fundamentals", {})
        peers = data.get("sector_peers", [])

        if not fund or not peers:
            return self.neutral(self.name, "No peer data for competitive comparison")

        # Metrics to compare
        target_metrics = {
            "profit_margin":    fund.get("profit_margin"),
            "return_on_equity": fund.get("return_on_equity"),
            "revenue_growth":   fund.get("revenue_growth"),
            "operating_margin": fund.get("operating_margin"),
        }

        if all(v is None for v in target_metrics.values()):
            return self.neutral(self.name, "No comparable metrics available")

        # Fetch peer metrics (cached for 24 hours)
        peer_data = [_get_peer_metrics(peer) for peer in peers[:6]]

        if not peer_data:
            return self.neutral(self.name, "Could not fetch peer data")

        score   = 50.0
        details = {}
        beats   = 0
        total   = 0

        for metric, target_val in target_metrics.items():
            if target_val is None:
                continue
            peer_vals = [p[metric] for p in peer_data if p.get(metric) is not None]
            if not peer_vals:
                continue
            peer_median = float(np.median(peer_vals))
            pct_rank    = sum(1 for v in peer_vals if target_val > v) / len(peer_vals)
            total += 1
            if pct_rank >= 0.67:
                beats += 1
                score += 8
                details[metric] = (f"Top-tier: {target_val:.2%} "
                                   f"(peer median {peer_median:.2%})")
            elif pct_rank <= 0.33:
                score -= 6
                details[metric] = (f"Lagging: {target_val:.2%} "
                                   f"(peer median {peer_median:.2%})")
            else:
                details[metric] = (f"Average: {target_val:.2%} "
                                   f"(peer median {peer_median:.2%})")

        # Rank verdict
        rank_pct = beats / total if total > 0 else 0.5
        if rank_pct >= 0.75:
            moat = "Strong competitive advantage"
        elif rank_pct >= 0.50:
            moat = "Moderate competitive advantage"
        else:
            moat = "Weak competitive position"
        details["competitive_moat"] = moat

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=60, sell_threshold=40)
        confidence = min(0.82, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":   details,
                "peers_compared": len(peer_data),
                "top_quartile_metrics": beats,
                "verdict":   f"Competitive rank score: {score:.0f}/100 — {moat}",
            }
        )

    # ── Crypto competitive analysis ───────────────────────────────────────────

    def _crypto_competitive(self, symbol: str, data: dict) -> TechniqueResult:
        """
        For crypto: compare 30-day return and volume vs peer universe.
        """
        hist = data.get("price_history")
        if hist is None or hist.empty:
            return self.neutral(self.name, "No crypto data for comparison")

        score   = 50.0
        details = {}

        close  = hist["Close"]
        volume = hist["Volume"]

        # 30-day return
        if len(close) >= 30:
            ret_30 = (close.iloc[-1] / close.iloc[-30] - 1)
            # Compare vs crypto peers in universe (use pre-loaded data if available)
            peer_returns = data.get("crypto_peer_returns", {})
            if peer_returns:
                median_ret = float(np.median(list(peer_returns.values())))
                if ret_30 > median_ret + 0.05:
                    score += 15; details["relative_return"] = f"Outperforming: {ret_30:.1%} vs peer {median_ret:.1%}"
                elif ret_30 < median_ret - 0.05:
                    score -= 12; details["relative_return"] = f"Underperforming: {ret_30:.1%} vs peer {median_ret:.1%}"
                else:
                    details["relative_return"] = f"In-line: {ret_30:.1%}"
            else:
                if ret_30 > 0.10:
                    score += 10; details["return_30d"] = f"Strong: {ret_30:.1%}"
                elif ret_30 < -0.15:
                    score -= 10; details["return_30d"] = f"Weak: {ret_30:.1%}"

        # Volume trend
        if len(volume) >= 30:
            vol_7  = volume.iloc[-7:].mean()
            vol_30 = volume.iloc[-30:].mean()
            if vol_7 > vol_30 * 1.5:
                score += 10; details["volume_dominance"] = "Volume surging — strong interest"
            elif vol_7 < vol_30 * 0.5:
                score -= 8;  details["volume_dominance"] = "Volume declining — losing interest"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=60, sell_threshold=40)
        confidence = min(0.75, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={"details": details,
                       "verdict": f"Crypto competitive score: {score:.0f}/100"}
        )
