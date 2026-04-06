"""
T3 — Bridgewater-Inspired Risk Analysis
Evaluates risk-adjusted return quality: Sharpe, drawdown, volatility, beta.
Applied to both stocks and crypto (crypto will naturally score lower on risk).
"""

import logging
import numpy as np
import pandas as pd
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_DAILY = 0.045 / TRADING_DAYS


class BridgewaterRisk(BaseTechnique):
    name = "T3_Risk"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        hist = data.get("price_history")
        if hist is None or len(hist) < 30:
            return self.neutral(self.name, "Insufficient price history for risk analysis")

        close   = hist["Close"].astype(float)
        returns = close.pct_change().dropna()

        # ── Annualised volatility ─────────────────────────────────────────────
        vol_daily = returns.std()
        vol_annual = vol_daily * np.sqrt(TRADING_DAYS)

        # ── Sharpe ratio ──────────────────────────────────────────────────────
        excess   = returns - RISK_FREE_DAILY
        sharpe   = (excess.mean() / vol_daily * np.sqrt(TRADING_DAYS)
                    if vol_daily > 0 else 0)

        # ── Max drawdown ──────────────────────────────────────────────────────
        roll_max   = close.cummax()
        drawdown   = (close - roll_max) / roll_max
        max_dd     = float(drawdown.min())

        # ── VaR (95%) ─────────────────────────────────────────────────────────
        var_95 = float(np.percentile(returns, 5))

        # ── Beta (vs SPY — approx using broad market returns if available) ───
        beta       = data.get("fundamentals", {}).get("beta") or 1.0

        # ── Scoring ───────────────────────────────────────────────────────────
        score = 50.0
        details = {}

        # Sharpe contribution
        if sharpe > 1.5:
            score += 20; details["sharpe"] = f"Excellent: {sharpe:.2f}"
        elif sharpe > 1.0:
            score += 12; details["sharpe"] = f"Good: {sharpe:.2f}"
        elif sharpe > 0.5:
            score += 5;  details["sharpe"] = f"Moderate: {sharpe:.2f}"
        elif sharpe < 0:
            score -= 15; details["sharpe"] = f"Negative: {sharpe:.2f}"
        else:
            details["sharpe"] = f"Weak: {sharpe:.2f}"

        # Volatility — penalise high vol
        if vol_annual < 0.20:
            score += 10; details["volatility"] = f"Low: {vol_annual:.1%}"
        elif vol_annual < 0.40:
            score += 3;  details["volatility"] = f"Moderate: {vol_annual:.1%}"
        elif vol_annual > 0.80:
            score -= 15; details["volatility"] = f"Very High: {vol_annual:.1%}"
        else:
            score -= 5;  details["volatility"] = f"High: {vol_annual:.1%}"

        # Max drawdown penalty
        if max_dd > -0.10:
            score += 10; details["max_drawdown"] = f"Shallow: {max_dd:.1%}"
        elif max_dd > -0.25:
            score += 3;  details["max_drawdown"] = f"Moderate: {max_dd:.1%}"
        elif max_dd < -0.50:
            score -= 15; details["max_drawdown"] = f"Severe: {max_dd:.1%}"
        else:
            score -= 5;  details["max_drawdown"] = f"Significant: {max_dd:.1%}"

        # Beta — moderate beta is ideal
        if 0.8 <= beta <= 1.3:
            score += 5;  details["beta"] = f"Market-like: {beta:.2f}"
        elif beta > 2.0:
            score -= 8;  details["beta"] = f"High beta: {beta:.2f}"
        elif beta < 0.3:
            score += 2;  details["beta"] = f"Low beta / defensive: {beta:.2f}"
        else:
            details["beta"] = f"Beta: {beta:.2f}"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=58, sell_threshold=40)
        confidence = min(0.85, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":      details,
                "sharpe":       round(sharpe, 3),
                "vol_annual":   f"{vol_annual:.1%}",
                "max_drawdown": f"{max_dd:.1%}",
                "var_95_daily": f"{var_95:.2%}",
                "beta":         round(beta, 2),
                "verdict":      f"Risk-adjusted quality score: {score:.0f}/100",
            }
        )
