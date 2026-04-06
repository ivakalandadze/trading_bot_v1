"""
T5 — BlackRock-Style Portfolio Fit Analysis
Evaluates whether adding this asset improves portfolio diversification and
risk-adjusted return, given the bot's current open positions.
"""

import logging
import numpy as np
import pandas as pd
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)

# Target sector/asset weights for a balanced portfolio
TARGET_ALLOCATION = {
    "Technology":             0.20,
    "Healthcare":             0.12,
    "Financial Services":     0.12,
    "Consumer Cyclical":      0.10,
    "Energy":                 0.08,
    "Consumer Defensive":     0.08,
    "Industrials":            0.08,
    "Utilities":              0.06,
    "Communication Services": 0.06,
    "Basic Materials":        0.05,
    "Real Estate":            0.05,
    "crypto":                 0.10,
}

MAX_SECTOR_WEIGHT = 0.30   # hard cap per sector


class BlackRockPortfolio(BaseTechnique):
    name = "T5_Portfolio"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        positions  = data.get("open_positions", [])   # list of position dicts
        fund       = data.get("fundamentals", {})
        hist       = data.get("price_history")
        score      = 50.0
        details    = {}

        # ── Asset type categorisation ─────────────────────────────────────────
        asset_type = "crypto" if is_crypto(symbol) else "stock"
        sector     = fund.get("sector", "Unknown") if not is_crypto(symbol) else "crypto"

        # ── Concentration check ───────────────────────────────────────────────
        if positions:
            # Count existing positions by sector
            sector_counts = {}
            total_pos = len(positions)
            for pos in positions:
                s = pos.get("sector", "Unknown")
                sector_counts[s] = sector_counts.get(s, 0) + 1

            existing_sector_pct = sector_counts.get(sector, 0) / max(total_pos, 1)
            target_pct          = TARGET_ALLOCATION.get(sector, 0.08)

            if existing_sector_pct > MAX_SECTOR_WEIGHT:
                score -= 20
                details["concentration"] = (f"Sector {sector} already at "
                                            f"{existing_sector_pct:.0%} — over limit")
            elif existing_sector_pct < target_pct:
                score += 15
                details["concentration"] = (f"Underweight {sector} "
                                            f"({existing_sector_pct:.0%} vs target {target_pct:.0%})")
            else:
                details["concentration"] = f"Sector {sector} at target weight"

            # Total position count check
            if total_pos >= 8:
                score -= 10
                details["position_count"] = f"Portfolio full: {total_pos} positions open"
            elif total_pos <= 3:
                score += 8
                details["position_count"] = f"Portfolio has room: {total_pos} positions"
        else:
            score += 10
            details["concentration"] = "Empty portfolio — any asset adds diversification"

        # ── Correlation check (if we have price history) ──────────────────────
        if hist is not None and positions:
            from data.market_data import get_price_history
            asset_returns = hist["Close"].pct_change().dropna()
            corr_scores = []

            for pos in positions[:5]:
                try:
                    pos_hist = get_price_history(pos["symbol"], period="3mo")
                    if pos_hist is not None and len(pos_hist) >= 20:
                        pos_returns = pos_hist["Close"].pct_change().dropna()
                        min_len = min(len(asset_returns), len(pos_returns))
                        if min_len >= 15:
                            corr = float(asset_returns.iloc[-min_len:].corr(
                                pos_returns.iloc[-min_len:]))
                            if not np.isnan(corr):
                                corr_scores.append(corr)
                except Exception:
                    pass

            if corr_scores:
                avg_corr = np.mean(corr_scores)
                if avg_corr < 0.3:
                    score += 12; details["avg_correlation"] = f"Low: {avg_corr:.2f} — diversifying"
                elif avg_corr > 0.7:
                    score -= 10; details["avg_correlation"] = f"High: {avg_corr:.2f} — redundant"
                else:
                    details["avg_correlation"] = f"Moderate: {avg_corr:.2f}"

        # ── Volatility fit ────────────────────────────────────────────────────
        if hist is not None:
            vol = hist["Close"].pct_change().std() * np.sqrt(252)
            if vol < 0.25:
                score += 5; details["volatility_fit"] = f"Stabilising: {vol:.1%} annual vol"
            elif vol > 0.80:
                score -= 8; details["volatility_fit"] = f"Destabilising: {vol:.1%} high vol"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=58, sell_threshold=40)
        confidence = min(0.80, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":    details,
                "sector":     sector,
                "asset_type": asset_type,
                "verdict":    f"Portfolio fit score: {score:.0f}/100",
            }
        )
