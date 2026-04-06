"""
T2 — Morgan Stanley-Style DCF Valuation
Calculates intrinsic value via discounted cash flow.
Not applicable to crypto — returns NEUTRAL for those assets.
"""

import logging
import numpy as np
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto, get_current_price

logger = logging.getLogger(__name__)

RISK_FREE_RATE_DEFAULT = 0.045   # fallback if macro data unavailable
EQUITY_PREMIUM  = 0.055          # historical equity risk premium
TERMINAL_GROWTH = 0.025          # long-run GDP growth for terminal value


class MorganStanleyDCF(BaseTechnique):
    name = "T2_DCF"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        if is_crypto(symbol):
            return self.neutral(self.name, "DCF not applicable to crypto assets")

        fund = data.get("fundamentals", {})
        if not fund:
            return self.neutral(self.name, "Insufficient fundamental data for DCF")

        fcf        = fund.get("free_cash_flow_ttm")
        market_cap = fund.get("market_cap")
        beta       = fund.get("beta") or 1.0
        rev_growth = fund.get("revenue_growth_5y") or fund.get("revenue_growth") or 0.05

        if not fcf or not market_cap or fcf <= 0:
            return self.neutral(self.name, "Negative or missing FCF — DCF unreliable")

        # ── Risk-free rate from live macro data (10-yr Treasury) ──────────────
        macro = data.get("macro", {})
        treasury_current = macro.get("treasury10", {}).get("current")
        risk_free_rate = (treasury_current / 100
                          if treasury_current and 1.0 < treasury_current < 15.0
                          else RISK_FREE_RATE_DEFAULT)

        # ── WACC estimate ─────────────────────────────────────────────────────
        cost_of_equity = risk_free_rate + beta * EQUITY_PREMIUM
        # Simplified: assume 70% equity, 30% debt at 5% cost (pre-tax)
        de         = fund.get("debt_to_equity") or 0.5
        debt_weight = de / (1 + de)
        equity_weight = 1 - debt_weight
        wacc = equity_weight * cost_of_equity + debt_weight * 0.05 * (1 - 0.21)
        wacc = max(0.06, min(0.20, wacc))   # clamp to realistic range

        # ── 5-year FCF projection ─────────────────────────────────────────────
        # Growth decays from current rate toward terminal growth over 5 years
        growth_rates = []
        for yr in range(1, 6):
            g = rev_growth * (1 - (yr - 1) * 0.15)   # fade by 15% each year
            g = max(TERMINAL_GROWTH, g)
            growth_rates.append(g)

        projected_fcf = []
        cf = fcf
        for g in growth_rates:
            cf = cf * (1 + g)
            projected_fcf.append(cf)

        # ── PV of projected FCF ───────────────────────────────────────────────
        pv_fcf = sum(cf / (1 + wacc) ** (i + 1)
                     for i, cf in enumerate(projected_fcf))

        # ── Terminal value (Gordon Growth) ────────────────────────────────────
        terminal_fcf = projected_fcf[-1] * (1 + TERMINAL_GROWTH)
        terminal_val = terminal_fcf / (wacc - TERMINAL_GROWTH)
        pv_terminal  = terminal_val / (1 + wacc) ** 5

        # ── Intrinsic value ───────────────────────────────────────────────────
        intrinsic_value = pv_fcf + pv_terminal

        margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # ── Score ─────────────────────────────────────────────────────────────
        # margin_of_safety > 0 = undervalued; < 0 = overvalued
        score = 50.0 + margin_of_safety * 100   # linear mapping ±50 pts at ±50% MoS
        score = self._clamp(score)

        # ── Sensitivity summary ───────────────────────────────────────────────
        verdict = ("Undervalued" if margin_of_safety > 0.20
                   else "Overvalued" if margin_of_safety < -0.20
                   else "Fairly Valued")

        signal     = self._score_to_signal(score, buy_threshold=62, sell_threshold=38)
        confidence = min(0.9, abs(margin_of_safety) * 2)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "fcf_ttm_B":          round(fcf / 1e9, 3),
                "market_cap_B":       round(market_cap / 1e9, 3),
                "intrinsic_value_B":  round(intrinsic_value / 1e9, 3),
                "margin_of_safety":   f"{margin_of_safety:.1%}",
                "wacc":               f"{wacc:.2%}",
                "risk_free_rate":     f"{risk_free_rate:.2%}",
                "rev_growth_assumed": f"{rev_growth:.1%}",
                "terminal_growth":    f"{TERMINAL_GROWTH:.1%}",
                "verdict":            verdict,
                "key_risk":           "Model assumes stable FCF growth; disruption could invalidate",
            }
        )
