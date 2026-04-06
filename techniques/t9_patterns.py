"""
T9 — Renaissance Technologies Pattern Finder
Identifies statistical edges: seasonal patterns, insider buying,
short squeeze potential, and institutional accumulation signals.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from .base_technique import BaseTechnique, TechniqueResult
from data.market_data import is_crypto

logger = logging.getLogger(__name__)


class RenaissancePatterns(BaseTechnique):
    name = "T9_Patterns"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        hist     = data.get("price_history")
        fund     = data.get("fundamentals", {})
        insider  = data.get("insider", {})

        if hist is None or len(hist) < 60:
            return self.neutral(self.name, "Insufficient history for pattern analysis")

        score   = 50.0
        details = {}
        close   = hist["Close"].astype(float)
        volume  = hist["Volume"].astype(float)

        # ── 1. Seasonal pattern (current month vs historical avg) ──────────────
        month_edge = self._seasonal_edge(close, hist.index)
        if month_edge is not None:
            details["seasonal_edge"] = month_edge["summary"]
            score += month_edge["score_adj"]

        # ── 2. Short interest (squeeze potential) ─────────────────────────────
        short_ratio = fund.get("short_ratio")        # days to cover
        short_pct   = fund.get("shares_short_pct")   # % of float

        if short_pct is not None:
            if short_pct > 0.20:
                score += 12   # high short interest = squeeze potential
                details["short_squeeze"] = (f"High short: {short_pct:.1%} float — "
                                             "squeeze potential")
            elif short_pct > 0.10:
                score += 5
                details["short_squeeze"] = f"Moderate short: {short_pct:.1%} float"
            else:
                details["short_squeeze"] = f"Low short: {short_pct:.1%} float"

        if short_ratio is not None:
            details["days_to_cover"] = f"{short_ratio:.1f} days"

        # ── 3. Insider buying sentiment ───────────────────────────────────────
        if not is_crypto(symbol):
            sentiment = insider.get("net_insider_sentiment", "unknown")
            net_val   = insider.get("net_insider_value", 0)
            if sentiment == "bullish":
                score += 12
                details["insider"] = f"Net buying: ${net_val/1e6:.1f}M net insider purchases"
            elif sentiment == "bearish":
                score -= 10
                details["insider"] = f"Net selling: ${abs(net_val)/1e6:.1f}M net insider sales"
            else:
                details["insider"] = "Neutral/unknown insider activity"

        # ── 4. Momentum anomaly (price acceleration) ──────────────────────────
        # 12-month momentum minus last month (Jegadeesh & Titman variant)
        if len(close) >= 252:
            ret_12m = close.iloc[-1] / close.iloc[-252] - 1
            ret_1m  = close.iloc[-1] / close.iloc[-21] - 1
            mom_adj = ret_12m - ret_1m   # remove reversal effect

            if mom_adj > 0.15:
                score += 10
                details["momentum_anomaly"] = f"Strong 12m-1m momentum: {mom_adj:.1%}"
            elif mom_adj < -0.15:
                score -= 8
                details["momentum_anomaly"] = f"Negative 12m-1m momentum: {mom_adj:.1%}"
            else:
                details["momentum_anomaly"] = f"Moderate momentum: {mom_adj:.1%}"

        # ── 5. Volume accumulation pattern ─────────────────────────────────────
        # On-Balance Volume (OBV) trend
        obv   = self._obv(close, volume)
        if len(obv) >= 20:
            obv_slope = (obv.iloc[-1] - obv.iloc[-20]) / max(abs(obv.iloc[-20]), 1)
            if obv_slope > 0.05:
                score += 8
                details["obv_trend"] = "OBV rising — institutional accumulation signal"
            elif obv_slope < -0.05:
                score -= 8
                details["obv_trend"] = "OBV falling — distribution signal"

        # ── 6. Price gap analysis ──────────────────────────────────────────────
        gaps = self._detect_gaps(hist)
        if gaps["gap_up_count"] > gaps["gap_down_count"]:
            score += 4
            details["gap_pattern"] = f"More up-gaps ({gaps['gap_up_count']}) than down ({gaps['gap_down_count']})"
        elif gaps["gap_down_count"] > gaps["gap_up_count"]:
            score -= 4
            details["gap_pattern"] = f"More down-gaps ({gaps['gap_down_count']}) — bearish"

        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=60, sell_threshold=40)
        confidence = min(0.82, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details": details,
                "verdict": f"Statistical edge score: {score:.0f}/100",
            }
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _seasonal_edge(self, close: pd.Series, index: pd.DatetimeIndex) -> dict | None:
        """Compute average return for current calendar month vs all months."""
        try:
            returns = close.pct_change().dropna()
            if returns.empty:
                return None
            months = pd.Series(returns.index.month, index=returns.index)
            df = pd.DataFrame({"return": returns.values, "month": months.values})
            monthly = df.groupby("month")["return"].mean()
            cur_month = datetime.now().month
            if cur_month not in monthly.index:
                return None

            cur_avg  = float(monthly[cur_month])
            all_avg  = float(monthly.mean())
            edge     = cur_avg - all_avg

            adj = edge * 200
            adj = max(-15, min(15, adj))

            return {
                "summary":   f"Month {cur_month} avg return: {cur_avg:.2%} (all months avg: {all_avg:.2%})",
                "score_adj": adj,
            }
        except Exception:
            return None

    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    def _detect_gaps(self, hist: pd.DataFrame) -> dict:
        """Count significant gap-ups and gap-downs in last 60 days."""
        df       = hist.tail(60)
        gap_up   = ((df["Open"] > df["Close"].shift(1) * 1.01)).sum()
        gap_down = ((df["Open"] < df["Close"].shift(1) * 0.99)).sum()
        return {"gap_up_count": int(gap_up), "gap_down_count": int(gap_down)}
