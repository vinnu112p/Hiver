"""
Phase 14: Safety & Escalation Policy Engine
Evaluates multiple orthogonal risk signals to decide AUTO_HANDLE vs ESCALATE:
- Classifier confidence
- Semantic retrieval similarity
- Category-level risk (Security, Fraud, Disputes)
- Keyword triggers (Explicit human request, Unauthorized charges)
- Ambiguity & Out-of-taxonomy detection
Always provides a clear, audit-ready justification reason.
"""
import re

# High-risk intents where automated action could cause data loss or security breach
ALWAYS_ESCALATE_INTENTS = {
    "account_access_security": "Potential account security compromise or credential takeover requires human verification."
}

# Regex for explicit human support requests
RE_HUMAN_REQUEST = re.compile(
    r"\b(human|representative|agent|person|manager|operator|supervisor|real person|talk to someone|speak to someone)\b",
    re.IGNORECASE
)

# Regex for financial dispute or unauthorized transactions
RE_FINANCIAL_DISPUTE = re.compile(
    r"\b(charged twice|double charge|unauthorized|stolen card|fraud|overcharged|dispute|refund my money|bank statement)\b",
    re.IGNORECASE
)

# Regex for urgent or legal threats
RE_LEGAL_OR_THREAT = re.compile(
    r"\b(lawyer|lawsuit|attorney|police|fraud|scam|sue you|legal action)\b",
    re.IGNORECASE
)

def evaluate_escalation(
    customer_message: str,
    predicted_intent: str,
    classifier_confidence: float,
    retrieval_similarity: float,
    min_intent_confidence: float = 0.35,
    min_retrieval_similarity: float = 0.45
) -> dict:
    """
    Evaluates whether a message is safe to auto-handle or must be escalated.
    Returns:
        {
            "decision": "auto_handle" | "escalate",
            "reason": str,
            "risk_level": "low" | "medium" | "high" | "critical",
            "signals": dict
        }
    """
    text = customer_message.strip()

    # Signal 1: Legal threat or scam allegations (Critical)
    if RE_LEGAL_OR_THREAT.search(text):
        return {
            "decision": "escalate",
            "reason": "Escalated due to legal escalation or fraud allegation.",
            "risk_level": "critical",
            "signals": {"trigger": "legal_or_threat"}
        }

    # Signal 2: Explicit request for a human agent
    if RE_HUMAN_REQUEST.search(text):
        return {
            "decision": "escalate",
            "reason": "Customer explicitly requested to communicate with a human agent.",
            "risk_level": "medium",
            "signals": {"trigger": "explicit_human_request"}
        }

    # Signal 3: Account security / credentials (High Risk)
    if predicted_intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "decision": "escalate",
            "reason": ALWAYS_ESCALATE_INTENTS[predicted_intent],
            "risk_level": "high",
            "signals": {"trigger": "account_security_intent"}
        }

    # Signal 4: Financial disputes & unauthorized transaction keywords
    if RE_FINANCIAL_DISPUTE.search(text):
        return {
            "decision": "escalate",
            "reason": "Disputed transaction or unauthorized charge requires billing team investigation.",
            "risk_level": "high",
            "signals": {"trigger": "financial_dispute_keywords"}
        }

    # Signal 5: Out of taxonomy or ambiguous
    if predicted_intent == "other_support":
        return {
            "decision": "escalate",
            "reason": "Customer query falls outside defined automation taxonomy.",
            "risk_level": "medium",
            "signals": {"trigger": "out_of_taxonomy"}
        }

    # Signal 6: Low classifier confidence
    if classifier_confidence < min_intent_confidence:
        return {
            "decision": "escalate",
            "reason": f"Low intent classification confidence ({classifier_confidence:.2f} < {min_intent_confidence:.2f}).",
            "risk_level": "medium",
            "signals": {"trigger": "low_classifier_confidence", "confidence": classifier_confidence}
        }

    # Signal 7: Low retrieval grounding similarity
    if retrieval_similarity < min_retrieval_similarity:
        return {
            "decision": "escalate",
            "reason": f"Weak historical precedent similarity ({retrieval_similarity:.2f} < {min_retrieval_similarity:.2f}).",
            "risk_level": "medium",
            "signals": {"trigger": "low_retrieval_similarity", "similarity": retrieval_similarity}
        }

    # Passed all risk checks: Safe for Auto-Handling
    return {
        "decision": "auto_handle",
        "reason": "High intent confidence, strong historical precedent, and low risk.",
        "risk_level": "low",
        "signals": {
            "classifier_confidence": classifier_confidence,
            "retrieval_similarity": retrieval_similarity
        }
    }
