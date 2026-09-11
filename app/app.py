"""
Phase 25: Runnable Interactive Demo UI
Minimal, functional Streamlit application demonstrating end-to-end grounded support:
- Intent Classification
- Multi-Signal Escalation Decision
- Grounded Reply Synthesis
- Transparent Historical Evidence Inspection
"""
import os
import sys
import json
import streamlit as st

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SupportAgentPipeline

st.set_page_config(page_title="Hiver AI Support Agent", layout="wide")

st.title("🎧 Hiver AI Support Agent — Spotify Support")
st.markdown(
    "A trustworthy, evidence-grounded AI customer support pipeline. "
    "Classifies intent, retrieves verified historical resolutions, synthesizes safe replies, and enforces multi-signal escalation."
)

@st.cache_resource
def load_pipeline():
    return SupportAgentPipeline()

pipeline = load_pipeline()

# Quick test query buttons
st.subheader("💡 Try Example Customer Inquiries:")
col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

default_msg = "I was charged twice for premium this month"
if col_ex1.button("💳 Double Charge (Dispute)"):
    default_msg = "I was charged twice for premium this month, please refund me!"
if col_ex2.button("🎵 Playback Pausing (FAQ)"):
    default_msg = "Songs keep stopping after 10 seconds on my iPhone"
if col_ex3.button("🚨 Compromised Account (Security)"):
    default_msg = "Someone hacked my account and changed the email address"
if col_ex4.button("💥 Desktop Crash (Technical)"):
    default_msg = "Spotify keeps crashing immediately when opened on Windows 10"

query_text = st.text_area("Enter Customer Message:", value=default_msg, height=90)
analyze_btn = st.button("🚀 Analyze Message", type="primary")

if analyze_btn or query_text:
    with st.spinner("Analyzing message through grounded pipeline..."):
        result = pipeline.process_message(query_text)

    st.divider()

    # Top Metric Banner
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Intent", result["intent"]["label"])
    col1.caption(f"Source: {result['intent']['source']}")

    col2.metric("Confidence", f"{result['intent']['confidence']*100:.1f}%")
    col3.metric("Decision", result["decision"].upper())
    col4.metric("Evidence Strength", f"{result['evidence_strength']*100:.1f}%")

    # Escalation Decision Banner
    if result["decision"] == "escalate":
        st.error(f"⚠️ **ACTION: ESCALATE TO HUMAN SPECIALIST**\n\n**Reason:** {result['reason']}")
    else:
        st.success(f"✅ **ACTION: AUTO-HANDLE**\n\n**Reason:** {result['reason']}")

    # Drafted Reply Section
    st.subheader("📝 Drafted Grounded Response")
    st.info(result["reply"])
    if result.get("grounding_summary"):
        st.caption(f"🔎 Grounding Context: {result['grounding_summary']}")

    # Historical Evidence Section
    st.subheader("📚 Retrieved Historical Support Precedents (Grounding Evidence)")
    if not result["retrieved_cases"]:
        st.warning("No historical cases met the minimum relevance threshold.")
    else:
        for i, case in enumerate(result["retrieved_cases"], 1):
            with st.expander(f"Precedent #{i} (Similarity: {case['similarity']:.2f}) — Case `{case['case_id']}`"):
                st.markdown(f"**Customer Asked:**\n> {case['customer']}")
                st.markdown(f"**Official Spotify Reply:**\n> {case['agent']}")
