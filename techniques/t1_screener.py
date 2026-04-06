"""
T1 — Goldman Sachs-Level Stock Screener
Evaluates fundamental quality: valuation, growth, profitability, balance sheet.
Crypto assets receive a simplified market-cap momentum screen.
"""

import logging
from .base_technique import BaseTechnique, TechniqueResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NEUTRAL
from data.market_data import is_crypto

logger = logging.getLogger(__name__)


class GoldmanScreener(BaseTechnique):
    name = "T1_Screener"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        if is_crypto(symbol):
            return self._crypto_screen(symbol, data)
        return self._stock_screen(symbol, data)

    # ── Stock screening ───────────────────────────────────────────────────────

    def _stock_screen(self, symbol: str, data: dict) -> TechniqueResult:
        fund = data.get("fundamentals", {})
        if not fund:
            return self.neutral(self.name, "No fundamental data available")

        score   = 50.0
        details = {}

        # 1. Valuation — P/E vs sector
        pe         = fund.get("pe_ratio")
        sector_pe  = data.get("sector_median_pe")
        if pe and pe > 0:
            if sector_pe:
                pe_ratio = pe / sector_pe
                if pe_ratio < 0.8:
                    score += 12   # trading at discount to peers
                    details["pe_vs_sector"] = f"Discount: {pe:.1f} vs sector {sector_pe:.1f}"
                elif pe_ratio > 1.5:
                    score -= 10   # premium valuation
                    details["pe_vs_sector"] = f"Premium: {pe:.1f} vs sector {sector_pe:.1f}"
                else:
                    details["pe_vs_sector"] = f"In-line: {pe:.1f} vs sector {sector_pe:.1f}"
            elif pe < 15:
                score += 8
                details["pe"] = f"Low absolute P/E: {pe:.1f}"
            elif pe > 40:
                score -= 8
                details["pe"] = f"High absolute P/E: {pe:.1f}"

        # 2. Revenue growth (5yr CAGR)
        rev_growth = fund.get("revenue_growth_5y") or fund.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > 0.20:
                score += 15; details["revenue_growth"] = f"Strong: {rev_growth:.1%}"
            elif rev_growth > 0.08:
                score += 8;  details["revenue_growth"] = f"Solid: {rev_growth:.1%}"
            elif rev_growth < 0:
                score -= 12; details["revenue_growth"] = f"Declining: {rev_growth:.1%}"
            else:
                details["revenue_growth"] = f"Modest: {rev_growth:.1%}"

        # 3. Debt/Equity health
        de = fund.get("debt_to_equity")
        if de is not None:
            if de < 0.5:
                score += 8;  details["debt_equity"] = f"Conservative: {de:.2f}"
            elif de > 2.0:
                score -= 10; details["debt_equity"] = f"Leveraged: {de:.2f}"
            else:
                details["debt_equity"] = f"Moderate: {de:.2f}"

        # 4. Profitability (ROE + margin)
        roe    = fund.get("return_on_equity")
        margin = fund.get("profit_margin")
        if roe is not None and roe > 0.15:
            score += 8; details["roe"] = f"High ROE: {roe:.1%}"
        elif roe is not None and roe < 0:
            score -= 8; details["roe"] = f"Negative ROE: {roe:.1%}"

        if margin is not None:
            if margin > 0.20:
                score += 7; details["profit_margin"] = f"High: {margin:.1%}"
            elif margin < 0:
                score -= 8; details["profit_margin"] = f"Negative: {margin:.1%}"

        # 5. Free Cash Flow positive
        fcf = fund.get("free_cash_flow_ttm")
        if fcf is not None:
            if fcf > 0:
                score += 5; details["fcf"] = f"Positive FCF: ${fcf/1e9:.2f}B"
            else:
                score -= 5; details["fcf"] = f"Negative FCF: ${fcf/1e9:.2f}B"

        # 6. Competitive moat proxy (high ROE + stable margins)
        moat = "weak"
        if roe and margin and roe > 0.20 and margin > 0.15:
            moat = "strong"; score += 5
        elif roe and margin and roe > 0.12 and margin > 0.08:
            moat = "moderate"; score += 2
        details["moat"] = moat

        score      = self._clamp(score)
        signal     = self._score_to_signal(score)
        confidence = min(0.9, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={
                "details": details,
                "sector":  fund.get("sector", "Unknown"),
                "industry": fund.get("industry", "Unknown"),
                "verdict": f"Fundamental quality score: {score:.0f}/100",
            }
        )

    # ── Crypto screening ──────────────────────────────────────────────────────

    def _crypto_screen(self, symbol: str, data: dict) -> TechniqueResult:
        """
        For crypto: use price momentum + volume trend as a proxy screener.
        """
        hist = data.get("price_history")
        if hist is None or hist.empty:
            return self.neutral(self.name, "No crypto price data")

        close  = hist["Close"]
        volume = hist["Volume"]
        score  = 50.0
        details = {}

        # 30-day momentum
        if len(close) >= 30:
            mom_30 = (close.iloc[-1] / close.iloc[-30] - 1) * 100
            if mom_30 > 20:
                score += 15; details["30d_momentum"] = f"+{mom_30:.1f}%"
            elif mom_30 > 5:
                score += 7;  details["30d_momentum"] = f"+{mom_30:.1f}%"
            elif mom_30 < -20:
                score -= 15; details["30d_momentum"] = f"{mom_30:.1f}%"
            elif mom_30 < -5:
                score -= 7;  details["30d_momentum"] = f"{mom_30:.1f}%"

        # Volume trend (7d avg vs 30d avg)
        if len(volume) >= 30:
            vol_7  = volume.iloc[-7:].mean()
            vol_30 = volume.iloc[-30:].mean()
            if vol_7 > vol_30 * 1.3:
                score += 10; details["volume_trend"] = "Rising (+30%)"
            elif vol_7 < vol_30 * 0.7:
                score -= 10; details["volume_trend"] = "Falling (-30%)"
            else:
                details["volume_trend"] = "Stable"

        # Price relative to 90-day high/low (range position)
        if len(close) >= 90:
            hi_90 = close.iloc[-90:].max()
            lo_90 = close.iloc[-90:].min()
            rng   = hi_90 - lo_90
            if rng > 0:
                pos = (close.iloc[-1] - lo_90) / rng
                details["range_position_90d"] = f"{pos:.0%} of 90d range"
                if pos > 0.8:
                    score -= 5    # near top, potential resistance
                elif pos < 0.3:
                    score += 5    # near lows, potential value

        score      = self._clamp(score)
        signal     = self._score_to_signal(score)
        confidence = min(0.75, abs(score - 50) / 50)

        return TechniqueResult(
            name=self.name, signal=signal,
            score=round(score, 1), confidence=round(confidence, 3),
            reasoning={"details": details, "asset_type": "crypto",
                       "verdict": f"Crypto momentum score: {score:.0f}/100"}
        )
