"""
Unit tests for unified end-to-end support pipeline.
"""
import pytest
from src.pipeline import SupportAgentPipeline

def test_pipeline_end_to_end_structure():
    pipeline = SupportAgentPipeline()
    res = pipeline.process_message("How do I cancel my subscription?")
    
    # Assert mandatory fields
    assert "message" in res
    assert "intent" in res
    assert "label" in res["intent"]
    assert "confidence" in res["intent"]
    assert "retrieved_cases" in res
    assert "reply" in res
    assert "decision" in res
    assert "reason" in res
    assert "evidence_strength" in res

    # Check types
    assert isinstance(res["retrieved_cases"], list)
    assert res["decision"] in ["auto_handle", "escalate"]
    assert isinstance(res["reply"], str)
    assert len(res["reply"]) > 0

def test_pipeline_handles_gibberish():
    pipeline = SupportAgentPipeline()
    res = pipeline.process_message("asdlkjf zxcvbnm qwertyuiop !!!")
    assert "decision" in res
    assert "reply" in res
    # Should safely escalate due to low confidence or other_support
    assert res["decision"] == "escalate"
