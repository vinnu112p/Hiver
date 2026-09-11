"""
Phase 13: Grounded Response Generator
Synthesizes customer support replies strictly grounded in retrieved historical evidence.
Guarantees:
- Never invents policies, refunds, actions, or URLs
- Follows brand support style (concise, polite, direct troubleshooting links)
- If evidence is weak or absent, signals low confidence and requests human escalation
- Dual mode: LLM synthesis if API key is provided; high-fidelity extractive grounding if offline
"""
import os
import json
import logging
from src.preprocess import clean_support_text

logger = logging.getLogger(__name__)

GROUNDING_SYSTEM_PROMPT = """You are an AI support assistant for Spotify.
Your job is to draft a helpful, concise customer response strictly grounded in the provided historical support examples.

CRITICAL RULES:
1. Grounding: Answer ONLY using the facts, troubleshooting steps, and policies present in the historical evidence.
2. No Hallucinations: NEVER invent refund amounts, promises of compensation, account actions, or arbitrary external links.
3. Style: Match official Spotify support style—friendly, concise (under 280 characters if possible), and direct.
4. Insufficient Evidence: If the historical cases do not contain a clear solution, recommend escalating to a human specialist.
5. Output Format: Return STRICT JSON ONLY with no extra markdown formatting:
{
  "reply": "<your drafted reply>",
  "grounding_summary": "<1 sentence explaining which historical evidence was used>",
  "confidence": <float between 0.0 and 1.0>
}
"""

class GroundedGenerator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def generate(self, customer_message: str, intent: str, retrieved_cases: list[dict]) -> dict:
        """
        Generates a grounded reply from historical support cases.
        Returns:
            {
                "reply": str,
                "grounding_summary": str,
                "confidence": float,
                "mode": "llm_synthesis" | "extractive_grounding" | "escalation_deflection"
            }
        """
        # Guard: No evidence retrieved or very low similarity
        if not retrieved_cases or (retrieved_cases[0].get("similarity", 0.0) < 0.35):
            return {
                "reply": "Thanks for reaching out. We want to take a closer look into this for you. Please hold on while we escalate your case to a human support agent.",
                "grounding_summary": "Insufficient historical evidence match; recommended for human review.",
                "confidence": 0.20,
                "mode": "escalation_deflection"
            }

        # Attempt LLM synthesis if API key is available
        if self.api_key:
            llm_result = self._generate_with_llm(customer_message, intent, retrieved_cases)
            if llm_result:
                return llm_result

        # Fallback: High-Fidelity Extractive Grounding
        return self._generate_extractive(customer_message, intent, retrieved_cases)

    def _generate_with_llm(self, customer_message: str, intent: str, retrieved_cases: list[dict]):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            evidence_bullets = []
            for i, c in enumerate(retrieved_cases, 1):
                evidence_bullets.append(
                    f"Example {i} (Similarity {c.get('similarity', 0.0):.2f}):\n"
                    f"Customer: {c.get('customer_message')}\n"
                    f"Official Spotify Reply: {c.get('historical_response')}\n"
                )
            evidence_str = "\n".join(evidence_bullets)

            user_prompt = (
                f"Customer Query: \"{customer_message}\"\n"
                f"Identified Intent: {intent}\n\n"
                f"Historical Resolved Cases (Your Evidence):\n"
                f"{evidence_str}\n\n"
                f"Draft a grounded response in JSON format:"
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=7.0
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            return {
                "reply": clean_support_text(data.get("reply", "")),
                "grounding_summary": str(data.get("grounding_summary", "Synthesized from historical cases.")),
                "confidence": round(float(data.get("confidence", 0.85)), 2),
                "mode": "llm_synthesis"
            }
        except Exception as e:
            logger.debug(f"LLM generation failed, falling back to extractive: {e}")
            return None

    def _generate_extractive(self, customer_message: str, intent: str, retrieved_cases: list[dict]) -> dict:
        """
        Extractive synthesis: extracts the verified resolution steps from the top historical match.
        Ensures 100% adherence to actual brand language with zero fabrication.
        """
        top_case = retrieved_cases[0]
        hist_resp = top_case.get("historical_response", "")
        sim = top_case.get("similarity", 0.70)

        summary = f"Grounded in verified historical resolution (case {top_case.get('case_id')}, similarity: {sim:.2f})"
        
        # Calculate grounding confidence as a function of retrieval similarity
        confidence = round(min(0.95, max(0.40, float(sim))), 2)

        return {
            "reply": clean_support_text(hist_resp),
            "grounding_summary": summary,
            "confidence": confidence,
            "mode": "extractive_grounding"
        }
