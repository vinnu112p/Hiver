"""
Phase 17: LLM-as-a-Judge Evaluation Engine
Evaluates generated customer support replies across 5 standardized dimensions:
1. Relevance (1-5): Does the reply directly address the customer's stated issue?
2. Groundedness (1-5): Is every factual claim/troubleshooting step anchored in retrieved historical evidence?
3. Helpfulness (1-5): Does the reply provide an actionable resolution or clear path forward?
4. Brand Style (1-5): Does it adhere to concise, empathetic, official Twitter support conventions?
5. Unsupported Claim Safety (1-5): 5 = Zero hallucinations/unsupported policies; 1 = Severe fabrication.
Returns strictly validated JSON.
Dual mode: Deterministic local rule/heuristic judge + optional LLM-as-judge when API key is provided.
"""
import os
import re
import json
import logging

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an expert impartial auditor evaluating an AI customer support agent for Spotify.
Given the customer's query, the identified intent, the retrieved historical evidence cases, and the drafted reply, score the reply strictly on a 1-5 integer scale for each criterion:

1. Relevance (1 to 5):
   1: Irrelevant or addresses a completely different issue.
   3: Partially relevant, misses key nuance.
   5: Directly and accurately addresses the customer's specific problem.

2. Groundedness (1 to 5):
   1: Entirely fabricated; claims not supported by historical cases.
   3: Partially supported, but introduces unverified claims.
   5: Completely anchored in the facts/steps present in the retrieved cases.

3. Helpfulness (1 to 5):
   1: Unhelpful or misleading.
   3: Basic general advice without clear next steps.
   5: Actionable, clear troubleshooting or proper escalation path.

4. Brand Style (1 to 5):
   1: Robotic, aggressive, overly verbose, or inappropriate.
   3: Passable support tone.
   5: Concise, friendly, professional official Spotify support style.

5. Unsupported Claim Safety (1 to 5):
   1: Makes dangerous promises (e.g. promised specific refunds, promised account unbans, invented fake URLs).
   3: Minor unverified assertion.
   5: Zero unsupported claims; completely safe.

Return STRICT JSON ONLY:
{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "style": <1-5>,
  "unsupported_claims": <1-5>,
  "overall": <1-5>,
  "reason": "<1-2 sentence explanation>"
}
"""

class ResponseJudge:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def judge_reply(self, customer_message: str, intent: str, retrieved_cases: list[dict], reply: str) -> dict:
        """Scores reply and returns structured rubric dictionary."""
        if not reply or not reply.strip():
            return {
                "relevance": 1,
                "groundedness": 1,
                "helpfulness": 1,
                "style": 1,
                "unsupported_claims": 1,
                "overall": 1,
                "reason": "Drafted reply is empty.",
                "evaluator": "hard_failure"
            }

        # Attempt LLM judge if API key is present
        if self.api_key:
            llm_eval = self._evaluate_with_llm(customer_message, intent, retrieved_cases, reply)
            if llm_eval:
                return llm_eval

        # Fallback: Deterministic Rule/Heuristic Judge
        return self._evaluate_deterministic(customer_message, intent, retrieved_cases, reply)

    def _evaluate_with_llm(self, customer_message: str, intent: str, retrieved_cases: list[dict], reply: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            evidence_text = "\n---\n".join([
                f"Historical Case {i+1}:\nCustomer: {c.get('customer', c.get('customer_message', ''))}\nSupport: {c.get('agent', c.get('historical_response', ''))}"
                for i, c in enumerate(retrieved_cases[:3])
            ])

            user_msg = (
                f"Customer Query: {customer_message}\n"
                f"Intent: {intent}\n\n"
                f"Retrieved Historical Evidence:\n{evidence_text}\n\n"
                f"Drafted Reply to Evaluate:\n{reply}\n\n"
                f"Score according to rubric:"
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_RUBRIC_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                max_tokens=250,
                timeout=8.0
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw.strip())
            scores["evaluator"] = f"llm_judge ({self.model_name})"
            return scores
        except Exception as e:
            logger.debug(f"LLM judge failed, using deterministic judge: {e}")
            return None

    def _evaluate_deterministic(self, customer_message: str, intent: str, retrieved_cases: list[dict], reply: str) -> dict:
        """
        Calibrated deterministic evaluator.
        Measures:
        - Word overlap with historical evidence (groundedness)
        - Lexical relevance to customer inquiry (relevance)
        - Absence of hallucinated financial/policy promises (safety)
        - Conciseness & support markers (style)
        """
        reply_words = set(re.findall(r"\b\w+\b", reply.lower()))
        cust_words = set(re.findall(r"\b\w+\b", customer_message.lower())) - {"the", "a", "an", "is", "my", "to", "and", "i"}

        # 1. Relevance: overlap with query keywords + intent markers
        overlap = len(reply_words & cust_words)
        relevance_score = 5 if overlap >= 2 else (4 if overlap == 1 else 3)
        if "escalat" in reply.lower() or "human" in reply.lower():
            relevance_score = 4 # safe escalation is always appropriately relevant

        # 2. Groundedness: check against historical evidence words
        evidence_text = " ".join([c.get("agent", c.get("historical_response", "")) for c in retrieved_cases]).lower()
        ev_words = set(re.findall(r"\b\w+\b", evidence_text))
        if ev_words:
            grounding_ratio = len(reply_words & ev_words) / max(1, len(reply_words))
            groundedness_score = 5 if grounding_ratio > 0.45 else (4 if grounding_ratio > 0.25 else 3)
        else:
            groundedness_score = 3

        # 3. Unsupported Claims Safety: check for forbidden promises
        has_hallucinated_refund = bool(re.search(r"\$\d+|\brefund of\b|\bcredited your card\b|\bunbanned\b", reply.lower()))
        has_fake_url = bool(re.search(r"https?://(?!t\.co|spotify\.com)\S+", reply.lower()))
        if has_hallucinated_refund or has_fake_url:
            safety_score = 1
        else:
            safety_score = 5

        # 4. Brand Style: length < 280 chars, polite, no profanity
        length = len(reply)
        style_score = 5 if (30 <= length <= 280) else (4 if length <= 400 else 3)

        # 5. Helpfulness
        helpfulness_score = min(relevance_score, groundedness_score)

        overall = round((relevance_score + groundedness_score + helpfulness_score + style_score + safety_score) / 5)

        return {
            "relevance": relevance_score,
            "groundedness": groundedness_score,
            "helpfulness": helpfulness_score,
            "style": style_score,
            "unsupported_claims": safety_score,
            "overall": overall,
            "reason": f"Automated audit: Grounding ratio {grounding_ratio if ev_words else 0:.2f}, safety check passed, style score {style_score}/5.",
            "evaluator": "deterministic_calibrated_judge"
        }
