"""
Unit tests for text preprocessing & quality filtering.
"""
import pytest
from src.preprocess import clean_customer_text, clean_support_text, is_valid_interaction

def test_clean_customer_text_removes_routing_mentions():
    raw = "@SpotifyCares @115887 My music stopped playing!"
    cleaned = clean_customer_text(raw)
    assert "@SpotifyCares" not in cleaned
    assert "@115887" not in cleaned
    assert "My music stopped playing!" in cleaned

def test_clean_customer_text_preserves_negations():
    raw = "@SpotifyCares I can't log in and it won't let me reset my password."
    cleaned = clean_customer_text(raw)
    assert "can't" in cleaned
    assert "won't" in cleaned

def test_clean_customer_text_unescapes_html():
    raw = "Rock &amp; Roll music is not loading &lt;help&gt;"
    cleaned = clean_customer_text(raw)
    assert "&" in cleaned
    assert "&amp;" not in cleaned

def test_is_valid_interaction_rejects_empty():
    assert is_valid_interaction("", "Hello")[0] is False
    assert is_valid_interaction("Help", "")[0] is False

def test_is_valid_interaction_rejects_too_short():
    assert is_valid_interaction("hi", "Hello how can we help?")[0] is False

def test_is_valid_interaction_accepts_valid():
    cust = "My playlist disappeared after the latest update."
    supp = "Could you DM us your account email so we can look backstage? /CK"
    is_valid, reason = is_valid_interaction(cust, supp)
    assert is_valid is True
    assert reason == "Valid"
