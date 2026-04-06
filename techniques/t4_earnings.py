"""
T4 — JPMorgan-Level Earnings Analysis
Evaluates EPS beat rate, earnings growth, and earnings momentum.
Returns NEUTRAL for crypto (earnings concept doesn't apply).
"""

import logging
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)


class JPMorganEarnings(BaseTechnique):
    name = "T4_Earnings"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        if is_crypto(symbol):
            return self.neutral(self.name, "Earnings analysis not applicable to crypto")

        earnings = data.get("earnings", {})
        fund     = data.get("fundamentals", {})

        if not earnings:
            return self.neutral(self.name, "No earnings data available")

        score   = 50.0
        details = {}

        # ── EPS beat rate ──────────────────────────────────────────────────────
        beat_rate = earnings.get("beat_rate")
        quarters  = earnings.get("quarters", 0)

        if beat_rate is not None and quarters >= 2:
            if beat_rate >= 0.75:
                score += 20; details["beat_rate"] = f"Excellent: {beat_rate:.0%} ({quarters} qtrs)"
            elif beat_rate >= 0.50:
                score += 10; details["beat_rate"] = f"Good: {beat_rate:.0%} ({quarters} qtrs)"
            elif beat_rate < 0.25:
                score -= 15; details["beat_rate"] = f"Poor: {beat_rate:.0%} ({quarters} qtrs)"
            else:
                details["beat_rate"] = f"Mixed: {beat_rate:.0%} ({quarters} qtrs)"
        else:
            details["beat_rate"] = "Insufficient data"

        # ── EPS growth ────────────────────────────────────────────────────────
        eps_growth = earnings.get("eps_growth")
        if eps_growth is not None:
            if eps_growth > 0.20:
                score += 15; details["eps_growth"] = f"Strong: {eps_growth:.1%}"
            elif eps_growth > 0.05:
                score += 7;  details["eps_growth"] = f"Positive: {eps_growth:.1%}"
            elif eps_growth < -0.10:
                score -= 12; details["eps_growth"] = f"Declining: {eps_growth:.1%}"
            else:
                details["eps_growth"] = f"Flat: {eps_growth:.1%}"

        # ── Revenue growth from fundamentals ──────────────────────────────────
        rev_growth = fund.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > 0.15:
                score += 10; details["revenue_growth_ttm"] = f"Strong: {rev_growth:.1%}"
            elif rev_growth > 0:
                score += 3;  details["revenue_growth_ttm"] = f"Positive: {rev_growth:.1%}"
            elif rev_growth < -0.05:
                score -= 8;  details["revenue_growth_ttm"] = f"Declining: {rev_growth:.1%}"

        # ── Most recent quarter check ─────────────────────────────────────────
        last_actual   = earnings.get("last_actual")
        last_estimate = earnings.get("last_estimate")
        if last_actual is not None and last_estimate is not None and last_estimate != 0:
            surprise_pct = (last_actual - last_estimate) / abs(last_estimate)
            if surprise_pct > 0.05:
                score += 8;  details["last_quarter"] = f"Beat by {surprise_pct:.1%}"
            elif surprise_pct < -0.05:
                score -= 10; details["last_quarter"] = f"Missed by {abs(surprise_pct):.1%}"
            else:
                details["last_quarter"] = f"In-line ({surprise_pct:+.1%})"

        # ── Earnings growth vs PE (PEG-like) ──────────────────────────────────
        peg = fund.get("peg_ratio")
        if peg is not None and peg > 0:
            if peg < 1.0:
                score += 8; details["peg_ratio"] = f"Attractive: {peg:.2f}"
            elif peg > 2.5:
                score -= 5; details["peg_ratio"] = f"Stretched: {peg:.2f}"
            else:
                details["peg_ratio"] = f"Fair: {peg:.2f}"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=62, sell_threshold=38)
        confidence = min(0.88, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details": details,
                "verdict": f"Earnings momentum score: {score:.0f}/100",
            }
        )
