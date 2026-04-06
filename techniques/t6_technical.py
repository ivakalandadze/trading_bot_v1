"""
T6 — Citadel-Grade Technical Analysis
Uses moving averages, RSI, MACD, Bollinger Bands, and volume analysis
to generate precise entry/exit signals. Works for both stocks and crypto.
"""

import logging
import numpy as np
import pandas as pd
from .base_technique import BaseTechnique, TechniqueResult

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(period).mean()
    loss    = (-delta.clip(upper=0)).rolling(period).mean()
    rs      = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period=20, num_std=2):
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


class CitadelTechnical(BaseTechnique):
    name = "T6_Technical"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        hist = data.get("price_history")
        if hist is None or len(hist) < 60:
            return self.neutral(self.name, "Insufficient price history for technical analysis")

        close  = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float) if "Volume" in hist.columns else None
        score  = 50.0
        details = {}
        signals_bull = 0
        signals_bear = 0

        n_bars = len(close)

        # ── Moving averages ───────────────────────────────────────────────────
        ma50  = close.rolling(50).mean()
        ma100 = close.rolling(100).mean() if n_bars >= 100 else None
        ma200 = close.rolling(200).mean() if n_bars >= 200 else None

        cur   = close.iloc[-1]

        if pd.notna(ma50.iloc[-1]):
            if cur > ma50.iloc[-1]:
                score += 8; signals_bull += 1
                details["ma50"] = f"Above MA50 ({ma50.iloc[-1]:.2f})"
            else:
                score -= 8; signals_bear += 1
                details["ma50"] = f"Below MA50 ({ma50.iloc[-1]:.2f})"

        # Golden/Death cross requires valid MA200 values for at least 2 bars
        if ma200 is not None and pd.notna(ma200.iloc[-1]) and pd.notna(ma200.iloc[-2]):
            ma50_prev  = ma50.iloc[-2]
            ma200_prev = ma200.iloc[-2]
            ma50_curr  = ma50.iloc[-1]
            ma200_curr = ma200.iloc[-1]
            if ma50_prev < ma200_prev and ma50_curr > ma200_curr:
                score += 15; signals_bull += 2
                details["cross"] = "Golden Cross detected (strong bullish)"
            elif ma50_prev > ma200_prev and ma50_curr < ma200_curr:
                score -= 15; signals_bear += 2
                details["cross"] = "Death Cross detected (strong bearish)"
            elif cur > ma200_curr:
                score += 5; signals_bull += 1
                details["ma200"] = f"Above MA200 ({ma200_curr:.2f}) — long-term uptrend"
            else:
                score -= 5; signals_bear += 1
                details["ma200"] = f"Below MA200 ({ma200_curr:.2f}) — long-term downtrend"
        elif n_bars < 200:
            details["ma200"] = f"Insufficient history ({n_bars} bars) for MA200 analysis"

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_series = _rsi(close)
        rsi        = float(rsi_series.iloc[-1])
        rsi_prev   = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi

        if rsi < 30:
            score += 15; signals_bull += 1
            details["rsi"] = f"Oversold: {rsi:.1f} — potential bounce"
        elif rsi < 45:
            score += 5; signals_bull += 1
            details["rsi"] = f"Soft: {rsi:.1f}"
        elif rsi > 70:
            score -= 12; signals_bear += 1
            details["rsi"] = f"Overbought: {rsi:.1f} — potential pullback"
        elif rsi > 60:
            score -= 3
            details["rsi"] = f"Hot: {rsi:.1f}"
        else:
            details["rsi"] = f"Neutral: {rsi:.1f}"

        # RSI momentum (direction of RSI)
        if rsi > rsi_prev + 2:
            score += 3; details["rsi_trend"] = "Rising"
        elif rsi < rsi_prev - 2:
            score -= 3; details["rsi_trend"] = "Falling"

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_line, signal_line, histogram = _macd(close)
        macd_val = float(macd_line.iloc[-1])
        sig_val  = float(signal_line.iloc[-1])
        hist_val = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2]) if len(histogram) >= 2 else hist_val

        if macd_val > sig_val:
            score += 8; signals_bull += 1
            details["macd"] = f"Bullish crossover (MACD {macd_val:.3f} > Signal {sig_val:.3f})"
        else:
            score -= 8; signals_bear += 1
            details["macd"] = f"Bearish (MACD {macd_val:.3f} < Signal {sig_val:.3f})"

        if hist_val > 0 and hist_val > hist_prev:
            score += 5; details["macd_momentum"] = "Histogram expanding bullish"
        elif hist_val < 0 and hist_val < hist_prev:
            score -= 5; details["macd_momentum"] = "Histogram expanding bearish"

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = _bollinger(close)
        bb_u = float(bb_upper.iloc[-1])
        bb_l = float(bb_lower.iloc[-1])
        bb_m = float(bb_mid.iloc[-1])

        if cur < bb_l:
            score += 10; signals_bull += 1
            details["bollinger"] = f"Below lower band ({bb_l:.2f}) — oversold"
        elif cur > bb_u:
            score -= 10; signals_bear += 1
            details["bollinger"] = f"Above upper band ({bb_u:.2f}) — overbought"
        else:
            bb_pct = (cur - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
            details["bollinger"] = f"Band position: {bb_pct:.0%} ({cur:.2f})"

        # ── Volume analysis ───────────────────────────────────────────────────
        if volume is not None and len(volume) >= 7:
            vol_7d  = volume.iloc[-7:].mean()
            vol_30d = volume.iloc[-30:].mean() if len(volume) >= 30 else vol_7d
            vol_ratio = vol_7d / vol_30d if vol_30d > 0 else 1.0

            price_up_today = close.iloc[-1] > close.iloc[-2]

            if vol_ratio > 1.5 and price_up_today:
                score += 8; details["volume"] = f"High volume breakout: {vol_ratio:.1f}x avg"
            elif vol_ratio > 1.5 and not price_up_today:
                score -= 8; details["volume"] = f"High volume selloff: {vol_ratio:.1f}x avg"
            elif vol_ratio < 0.6:
                score -= 3; details["volume"] = f"Low volume: {vol_ratio:.1f}x avg — weak conviction"
            else:
                details["volume"] = f"Volume normal: {vol_ratio:.1f}x avg"

        # ── Final score & confidence ──────────────────────────────────────────
        score      = self._clamp(score)
        signal     = self._score_to_signal(score, buy_threshold=60, sell_threshold=40)
        total_signals = signals_bull + signals_bear
        confidence = min(0.92, abs(score - 50) / 50 * (1 + 0.1 * total_signals))

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details":      details,
                "rsi":          round(rsi, 1),
                "macd":         round(macd_val, 4),
                "current_price": round(cur, 4),
                "bull_signals": signals_bull,
                "bear_signals": signals_bear,
                "verdict":      f"Technical score: {score:.0f}/100",
            }
        )
