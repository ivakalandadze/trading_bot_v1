"""
T10 — McKinsey-Level Macro Impact Assessment
Evaluates how the current macroeconomic environment (VIX, rates, dollar,
market trend) affects the attractiveness of a given asset.
"""

import logging
import numpy as np
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)


class McKinseyMacro(BaseTechnique):
    name = "T10_Macro"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        macro = data.get("macro", {})
        fund  = data.get("fundamentals", {})

        if not macro:
            return self.neutral(self.name, "No macro data available")

        score   = 50.0
        details = {}

        # ── Market fear gauge (VIX) ───────────────────────────────────────────
        vix_data = macro.get("vix", {})
        vix      = vix_data.get("current")
        if vix is not None:
            if vix < 15:
                score += 10; details["vix"] = f"Low fear ({vix:.1f}) — risk-on environment"
            elif vix < 25:
                score += 3;  details["vix"] = f"Moderate ({vix:.1f}) — normal market"
            elif vix < 35:
                score -= 10; details["vix"] = f"Elevated ({vix:.1f}) — caution advised"
            else:
                score -= 18; details["vix"] = f"Panic zone ({vix:.1f}) — extreme caution"

        # ── Interest rate environment ─────────────────────────────────────────
        rate_data = macro.get("treasury10", {})
        rate      = rate_data.get("current")
        rate_chg  = rate_data.get("1m_change")
        if rate is not None:
            sector = fund.get("sector", "") if not is_crypto(symbol) else ""

            if rate > 5.0:
                score -= 12; details["rates"] = f"10yr at {rate:.2f}% — headwind for growth stocks"
                if sector in ("Real Estate", "Utilities"):
                    score -= 5   # extra pain for rate-sensitive sectors
            elif rate < 3.5:
                score += 10; details["rates"] = f"10yr at {rate:.2f}% — supportive for risk assets"
            else:
                details["rates"] = f"10yr at {rate:.2f}% — neutral"

            if rate_chg is not None:
                if rate_chg > 10:   # rates rising fast (% change in yield level)
                    score -= 7; details["rate_trend"] = f"Rates rising sharply (+{rate_chg:.1f}%)"
                elif rate_chg < -10:
                    score += 7; details["rate_trend"] = f"Rates falling — bond rally signal"

        # ── US Dollar strength (DXY) ──────────────────────────────────────────
        dxy_data  = macro.get("dxy", {})
        dxy_chg   = dxy_data.get("1m_change")
        if dxy_chg is not None:
            if is_crypto(symbol):
                # Strong dollar typically hurts crypto
                if dxy_chg > 2:
                    score -= 8; details["dollar"] = f"Dollar strengthening ({dxy_chg:+.1f}%) — crypto headwind"
                elif dxy_chg < -2:
                    score += 8; details["dollar"] = f"Dollar weakening ({dxy_chg:+.1f}%) — crypto tailwind"
            else:
                # International exposure companies hurt by strong dollar
                if dxy_chg > 2:
                    score -= 4; details["dollar"] = f"Dollar strengthening ({dxy_chg:+.1f}%) — potential FX headwind"
                elif dxy_chg < -2:
                    score += 4; details["dollar"] = f"Dollar weakening ({dxy_chg:+.1f}%) — FX tailwind"

        # ── Broad market trend (SPY) ──────────────────────────────────────────
        spy_data  = macro.get("spy", {})
        spy_1m    = spy_data.get("1m_change")
        spy_above = spy_data.get("above_50ma")

        if spy_above is True:
            score += 8; details["market_trend"] = "SPY above 50-MA — bull market condition"
        elif spy_above is False:
            score -= 10; details["market_trend"] = "SPY below 50-MA — bear/correction"

        if spy_1m is not None:
            if spy_1m > 5:
                score += 5; details["spy_momentum"] = f"Market up {spy_1m:.1f}% (1m)"
            elif spy_1m < -5:
                score -= 5; details["spy_momentum"] = f"Market down {spy_1m:.1f}% (1m)"

        # ── Sector rotation signal ────────────────────────────────────────────
        sector = fund.get("sector", "") if not is_crypto(symbol) else "crypto"
        rotation_adj = self._sector_rotation_score(sector, vix, rate)
        if rotation_adj != 0:
            score += rotation_adj
            cycle = "late" if rotation_adj < 0 else "mid/early"
            details["sector_rotation"] = (f"{sector} — {'+' if rotation_adj>0 else ''}"
                                           f"{rotation_adj:.0f}pt cycle adjustment ({cycle} cycle)")

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=58, sell_threshold=42)
        confidence = min(0.85, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":   details,
                "vix":       vix,
                "rate_10yr": rate,
                "verdict":   f"Macro tailwind score: {score:.0f}/100",
            }
        )

    def _sector_rotation_score(self, sector: str,
                                vix: float | None,
                                rate: float | None) -> float:
        """
        Apply sector-rotation logic based on rate and fear environment.
        High rates → favour Energy, Financials, Staples over Tech/Growth.
        High VIX   → favour Staples, Healthcare, Utilities.
        """
        adj = 0.0

        high_rate = rate is not None and rate > 4.5
        low_rate  = rate is not None and rate < 3.5
        high_fear = vix  is not None and vix > 25

        favourites_high_rate  = {"Energy", "Financial Services", "Consumer Defensive", "Healthcare"}
        unfavoured_high_rate  = {"Technology", "Real Estate", "Utilities", "Consumer Cyclical"}
        defensive_sectors     = {"Consumer Defensive", "Healthcare", "Utilities"}

        if high_rate:
            if sector in favourites_high_rate:
                adj += 8
            elif sector in unfavoured_high_rate:
                adj -= 8

        if low_rate:
            if sector in ("Technology", "Consumer Cyclical"):
                adj += 6
            elif sector == "Financial Services":
                adj -= 4

        if high_fear:
            if sector in defensive_sectors:
                adj += 6
            elif sector in {"Technology", "Consumer Cyclical", "crypto"}:
                adj -= 6

        return adj
