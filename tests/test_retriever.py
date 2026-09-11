"""
Unit tests for FAISS semantic retrieval.
"""
import pytest
from src.retriever import SemanticRetriever

def test_retriever_search_returns_cases():
    retriever = SemanticRetriever()
    retriever.load_index()
    results = retriever.search("Music stops playing", intent="audio_playback_issue", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
    first = results[0]
    assert "case_id" in first
    assert "customer_message" in first
    assert "historical_response" in first
    assert "similarity" in first
    assert 0.0 <= first["similarity"] <= 1.0

def test_retriever_intent_prioritization():
    retriever = SemanticRetriever()
    results = retriever.search("I want to cancel my account", intent="cancellation_refund", top_k=2)
    assert len(results) > 0
    # Top result should match intent or have high similarity
    assert results[0]["similarity"] > 0.4
