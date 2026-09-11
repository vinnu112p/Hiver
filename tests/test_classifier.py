"""
Unit tests for intent classification and fallbacks.
"""
import pytest
from src.classifier import IntentClassifier, TAXONOMY_INTENTS

def test_classifier_initialization():
    clf = IntentClassifier()
    assert clf is not None
    assert len(clf.labels) == 7

def test_classifier_handles_empty_string():
    clf = IntentClassifier()
    res = clf.predict("")
    assert res["intent"] == "other_support"
    assert res["confidence"] == 1.0

def test_classifier_predicts_common_intents():
    clf = IntentClassifier()
    
    # Billing query
    res_bill = clf.predict("I was charged twice for premium subscription")
    assert res_bill["intent"] in ["billing_subscription", "cancellation_refund"]
    assert 0.0 <= res_bill["confidence"] <= 1.0

    # Playback query
    res_play = clf.predict("Every song keeps skipping and pausing on my phone")
    assert res_play["intent"] == "audio_playback_issue"

def test_classifier_structured_output():
    clf = IntentClassifier()
    res = clf.predict("How do I recover a deleted playlist?")
    assert "intent" in res
    assert "confidence" in res
    assert "source" in res
    assert "probabilities" in res
    assert res["intent"] in TAXONOMY_INTENTS
