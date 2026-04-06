"""
T7 — Harvard Endowment-Inspired Dividend Strategy
Evaluates dividend sustainability, growth, and income quality.
Returns NEUTRAL for crypto and non-dividend stocks.
"""

import logging
import yfinance as yf
import pandas as pd
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)


class HarvardDividend(BaseTechnique):
    name = "T7_Dividend"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        if is_crypto(symbol):
            return self.neutral(self.name, "Dividends not applicable to crypto")

        fund = data.get("fundamentals", {})

        div_yield    = fund.get("dividend_yield")
        payout_ratio = fund.get("payout_ratio")

        # Skip non-dividend stocks — not penalise them
        if not div_yield or div_yield < 0.005:
            return self.neutral(self.name, "No meaningful dividend — not a dividend stock")

        score   = 50.0
        details = {}

        # ── Yield quality ─────────────────────────────────────────────────────
        if div_yield > 0.06:
            score += 10; details["yield"] = f"High: {div_yield:.2%} — check sustainability"
        elif div_yield > 0.03:
            score += 18; details["yield"] = f"Attractive: {div_yield:.2%}"
        elif div_yield > 0.015:
            score += 8;  details["yield"] = f"Modest: {div_yield:.2%}"
        else:
            details["yield"] = f"Low: {div_yield:.2%}"

        # ── Payout ratio sustainability ────────────────────────────────────────
        if payout_ratio is not None and 0 < payout_ratio < 5:
            if payout_ratio < 0.40:
                score += 15; details["payout_ratio"] = f"Very sustainable: {payout_ratio:.0%}"
            elif payout_ratio < 0.60:
                score += 10; details["payout_ratio"] = f"Sustainable: {payout_ratio:.0%}"
            elif payout_ratio < 0.80:
                score += 3;  details["payout_ratio"] = f"Borderline: {payout_ratio:.0%}"
            else:
                score -= 15; details["payout_ratio"] = f"Unsustainable: {payout_ratio:.0%}"
        else:
            details["payout_ratio"] = "N/A"

        # ── Dividend growth history ────────────────────────────────────────────
        div_growth = self._calc_dividend_growth(symbol)
        if div_growth is not None:
            if div_growth > 0.08:
                score += 12; details["div_growth_5y"] = f"Strong CAGR: {div_growth:.1%}"
            elif div_growth > 0.03:
                score += 7;  details["div_growth_5y"] = f"Solid CAGR: {div_growth:.1%}"
            elif div_growth < 0:
                score -= 15; details["div_growth_5y"] = f"Cuts detected: {div_growth:.1%}"
            else:
                details["div_growth_5y"] = f"Flat: {div_growth:.1%}"

        # ── FCF coverage ──────────────────────────────────────────────────────
        fcf       = fund.get("free_cash_flow_ttm")
        mkt_cap   = fund.get("market_cap")
        if fcf and mkt_cap:
            fcf_yield = fcf / mkt_cap
            if fcf_yield > div_yield * 1.5:
                score += 8; details["fcf_coverage"] = f"FCF yield {fcf_yield:.2%} covers dividend well"
            elif fcf_yield < div_yield:
                score -= 10; details["fcf_coverage"] = "FCF yield below dividend yield — risk"

        # ── Return on equity (quality proxy) ─────────────────────────────────
        roe = fund.get("return_on_equity")
        if roe and roe > 0.15:
            score += 5; details["roe"] = f"Quality operator: ROE {roe:.1%}"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=65, sell_threshold=40)
        confidence = min(0.88, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":    details,
                "div_yield":  f"{div_yield:.2%}",
                "verdict":    f"Dividend quality score: {score:.0f}/100",
            }
        )

    def _calc_dividend_growth(self, symbol: str) -> float | None:
        """Calculate 5-year dividend CAGR from yfinance dividend history."""
        try:
            ticker    = yf.Ticker(symbol)
            divs      = ticker.dividends
            if divs is None or len(divs) < 4:
                return None
            # Annual aggregation
            annual = divs.resample("YE").sum()
            if len(annual) < 2:
                return None
            first = float(annual.iloc[0])
            last  = float(annual.iloc[-1])
            years = len(annual) - 1
            if first <= 0 or last <= 0:
                return None
            return (last / first) ** (1 / years) - 1
        except Exception:
            return None
