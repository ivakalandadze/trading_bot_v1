"""
llm_judge.py — Claude AI final decision gate.

After the signal engine votes 3+/10 on a direction, the LLM judge receives
a structured briefing of all technique results and renders a final
GO / NO-GO with a reasoning explanation.
This is the hybrid intelligence layer — algorithms find the signal,
Claude provides the judgment.
"""

import json
import logging
from dataclasses import dataclass

import anthropic

import config
from signal_engine import TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    approved:   bool
    action:     str     # 'BUY' | 'SELL' | 'SKIP'
    reasoning:  str
    confidence: float   # LLM's own stated confidence 0-1
    risk_notes: str


class LLMJudge:
    """
    Calls Claude claude-sonnet-4-6 to make the final trade decision.
    Falls back to approving the signal if the API key is missing.
    """

    def __init__(self):
        if config.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            self.client = None
            logger.warning("No ANTHROPIC_API_KEY — LLM judge disabled, auto-approving signals")

    def judge(self, signal: TradingSignal,
              current_price: float,
              stop_loss: float,
              take_profit: float) -> JudgmentResult:
        """
        Submit a trade signal for LLM review.
        Returns JudgmentResult with approved=True/False.
        """
        if self.client is None:
            return JudgmentResult(
                approved=True, action=signal.direction,
                reasoning="LLM judge bypassed (no API key configured)",
                confidence=0.6, risk_notes="Auto-approved"
            )

        prompt = self._build_prompt(signal, current_price, stop_loss, take_profit)

        try:
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text
            return self._parse_response(raw, signal.direction)

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return JudgmentResult(
                approved=False, action="SKIP",
                reasoning=f"API error — trade rejected for safety: {str(e)[:100]}",
                confidence=0.0, risk_notes="API failure — never auto-approve"
            )
        except Exception as e:
            logger.error(f"LLM judge unexpected error: {e}")
            return JudgmentResult(
                approved=False, action="SKIP",
                reasoning=f"Judge failed: {str(e)[:100]}",
                confidence=0.0, risk_notes="Error"
            )

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(self, signal: TradingSignal,
                      current_price: float,
                      stop_loss: float,
                      take_profit: float) -> str:
        buy_count  = signal.buy_count
        sell_count = signal.sell_count
        total      = len(signal.technique_results)

        # Technique results summary
        tech_lines = []
        for name, result in signal.techniques_summary.items():
            if result["applicable"]:
                tech_lines.append(
                    f"  • {name}: {result['signal']} "
                    f"(score={result['score']:.0f}/100, "
                    f"conf={result['confidence']:.0%}) — {result['verdict']}"
                )
        tech_summary = "\n".join(tech_lines) or "  No applicable techniques"

        rr_ratio = abs(take_profit - current_price) / max(abs(current_price - stop_loss), 0.0001)

        prompt = f"""You are an expert autonomous trading system risk manager.

## Trade Signal Summary

**Symbol**: {signal.symbol} ({signal.asset_type.upper()})
**Proposed Action**: {signal.direction}
**Current Price**: ${current_price:.4f}
**Stop-Loss**: ${stop_loss:.4f} ({abs(current_price - stop_loss)/current_price*100:.1f}% from entry)
**Take-Profit**: ${take_profit:.4f} ({abs(take_profit - current_price)/current_price*100:.1f}% from entry)
**Risk/Reward Ratio**: {rr_ratio:.2f}:1

## Analysis Technique Votes
{buy_count} BUY / {sell_count} SELL / {total - buy_count - sell_count} NEUTRAL
Weighted Aggregate Score: {signal.score:.1f}/100

### Individual Technique Results:
{tech_summary}

## Your Task
Review this trade signal and decide: should the bot execute this trade?

Consider:
1. Is there genuine multi-factor confirmation or are the signals correlated/overlapping?
2. Does the risk/reward ratio justify the trade?
3. Are there any red flags in the technique results (conflicts, low confidence)?
4. Is this a high-quality setup or a marginal one to skip?

Respond in this EXACT JSON format (no markdown, no extra text):
{{
  "action": "BUY" or "SELL" or "SKIP",
  "approved": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence explanation of your decision",
  "risk_notes": "Key risks or concerns to watch"
}}"""

        return prompt

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str, proposed_action: str) -> JudgmentResult:
        """Parse JSON response from Claude."""
        try:
            # Strip any accidental markdown fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines   = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            data = json.loads(cleaned)

            action   = data.get("action", "SKIP").upper()
            approved = bool(data.get("approved", False))
            conf     = float(data.get("confidence", 0.5))
            reason   = str(data.get("reasoning", ""))
            risks    = str(data.get("risk_notes", ""))

            # Safety: if approved=True but action=SKIP, treat as not approved
            if action == "SKIP":
                approved = False

            return JudgmentResult(
                approved=approved, action=action,
                reasoning=reason, confidence=conf,
                risk_notes=risks,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"LLM response parse error: {e} | raw={raw[:200]}")
            # Conservative fallback on parse failure
            return JudgmentResult(
                approved=False, action="SKIP",
                reasoning=f"Could not parse LLM response: {raw[:100]}",
                confidence=0.0, risk_notes="Parse error"
            )
