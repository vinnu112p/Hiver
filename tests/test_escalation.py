"""
Unit tests for escalation safety engine.
"""
import pytest
from src.escalation import evaluate_escalation

def test_escalation_security_intent_always_escalates():
    res = evaluate_escalation(
        customer_message="Someone changed my password and email",
        predicted_intent="account_access_security",
        classifier_confidence=0.95,
        retrieval_similarity=0.88
    )
    assert res["decision"] == "escalate"
    assert "security" in res["reason"].lower() or "credential" in res["reason"].lower()

def test_escalation_explicit_human_request():
    res = evaluate_escalation(
        customer_message="Let me speak to a human manager please",
        predicted_intent="other_support",
        classifier_confidence=0.85,
        retrieval_similarity=0.75
    )
    assert res["decision"] == "escalate"
    assert "human" in res["reason"].lower()

def test_escalation_disputed_charge():
    res = evaluate_escalation(
        customer_message="I was charged twice on my card, refund my money!",
        predicted_intent="billing_subscription",
        classifier_confidence=0.90,
        retrieval_similarity=0.82
    )
    assert res["decision"] == "escalate"
    assert "dispute" in res["reason"].lower() or "charge" in res["reason"].lower()

def test_escalation_low_confidence_escalates():
    res = evaluate_escalation(
        customer_message="It did a weird thing earlier",
        predicted_intent="audio_playback_issue",
        classifier_confidence=0.20, # Very low confidence
        retrieval_similarity=0.70
    )
    assert res["decision"] == "escalate"

def test_escalation_safe_playback_auto_handles():
    res = evaluate_escalation(
        customer_message="Songs keep pausing after 10 seconds on my iPhone",
        predicted_intent="audio_playback_issue",
        classifier_confidence=0.85,
        retrieval_similarity=0.80
    )
    assert res["decision"] == "auto_handle"
    assert res["risk_level"] == "low"
