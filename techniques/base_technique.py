"""
techniques/base_technique.py — Shared contract for all analysis techniques.

Every technique must:
  1. Accept a symbol and pre-fetched data dict
  2. Return a TechniqueResult with signal, score, confidence, and reasoning
"""

from dataclasses import dataclass, field
from typing import Optional


SIGNAL_BUY     = "BUY"
SIGNAL_SELL    = "SELL"
SIGNAL_NEUTRAL = "NEUTRAL"


@dataclass
class TechniqueResult:
    name:       str
    signal:     str                   # BUY | SELL | NEUTRAL
    score:      float                 # 0–100  (higher = more bullish)
    confidence: float                 # 0–1
    reasoning:  dict = field(default_factory=dict)
    applicable: bool = True           # False if technique doesn't apply (e.g. DCF for crypto)

    @property
    def is_buy(self)  -> bool: return self.signal == SIGNAL_BUY
    @property
    def is_sell(self) -> bool: return self.signal == SIGNAL_SELL

    def summary_line(self) -> str:
        return (f"[{self.name}] {self.signal} | score={self.score:.1f} "
                f"confidence={self.confidence:.0%}")


class BaseTechnique:
    """Abstract base — subclasses override analyse()."""

    name: str = "BaseTechnique"

    def analyse(self, symbol: str, data: dict) -> TechniqueResult:
        raise NotImplementedError

    @staticmethod
    def neutral(name: str, reason: str = "Not applicable") -> TechniqueResult:
        return TechniqueResult(
            name=name, signal=SIGNAL_NEUTRAL,
            score=50.0, confidence=0.0,
            reasoning={"reason": reason},
            applicable=False,
        )

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _score_to_signal(score: float,
                         buy_threshold: float  = 60.0,
                         sell_threshold: float = 40.0) -> str:
        if score >= buy_threshold:
            return SIGNAL_BUY
        if score <= sell_threshold:
            return SIGNAL_SELL
        return SIGNAL_NEUTRAL
