"""
Phase 8: Interactive Streamlit Labeler Tool
Provides an intuitive, web-based UI for manual inspection, labeling, and auditing
of the 200 Golden Evaluation Set cases.
"""
import os
import sys
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hiver Golden Set Labeler", layout="wide")

GOLDEN_CSV = os.path.join("data", "golden", "golden_set.csv")
CANDIDATES_CSV = os.path.join("data", "golden", "candidates_to_label.csv")

INTENTS = [
    "billing_subscription",
    "cancellation_refund",
    "account_access_security",
    "audio_playback_issue",
    "app_crash_technical",
    "library_playlist_content",
    "other_support"
]

DECISIONS = ["auto_handle", "escalate"]
DIFFICULTIES = ["normal", "ambiguous", "hard"]

st.title("🎯 Hiver AI Support — Golden Set Annotation & Audit Tool")
st.markdown("Use this tool to manually inspect, label, and audit customer tweets for ground-truth evaluation.")

# Load datasets
if not os.path.exists(CANDIDATES_CSV):
    st.error(f"Candidate file not found at {CANDIDATES_CSV}. Run scripts/sample_golden_candidates.py first.")
    st.stop()

df_cand = pd.read_csv(CANDIDATES_CSV)

if os.path.exists(GOLDEN_CSV):
    df_golden = pd.read_csv(GOLDEN_CSV)
else:
    df_golden = pd.DataFrame(columns=["id", "text", "intent", "expected_decision", "difficulty", "notes"])

# Progress metrics
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Candidates", len(df_cand))
col_m2.metric("Completed Labels", len(df_golden))
col_m3.metric("Remaining", max(0, len(df_cand) - len(df_golden)))

st.divider()

# Select case index to inspect / annotate
idx = st.number_input("Case Index (0-199)", min_value=0, max_value=len(df_cand)-1, value=min(len(df_golden), len(df_cand)-1))

case_row = df_cand.iloc[idx]
case_id = str(case_row["id"])
text = str(case_row["text"])
ref_response = str(case_row.get("support_response_reference", ""))
tag = str(case_row.get("category_tag", ""))
suggested_diff = str(case_row.get("target_difficulty", "normal"))

# Check if already labeled
existing_label = df_golden[df_golden["id"].astype(str) == case_id]
is_already_labeled = len(existing_label) > 0

st.subheader(f"Case #{idx} — ID: `{case_id}`")
st.info(f"**Customer Message:**\n\n\"{text}\"")

if ref_response and ref_response != 'nan':
    with st.expander("Show Historical Brand Reply Reference"):
        st.write(ref_response)

with st.form("label_form"):
    col1, col2, col3 = st.columns(3)

    default_intent = existing_label.iloc[0]["intent"] if is_already_labeled else INTENTS[0]
    default_dec = existing_label.iloc[0]["expected_decision"] if is_already_labeled else ("escalate" if suggested_diff == "hard" else "auto_handle")
    default_diff = existing_label.iloc[0]["difficulty"] if is_already_labeled else suggested_diff
    default_notes = existing_label.iloc[0]["notes"] if is_already_labeled else f"Stratified sample: {tag}"

    intent = col1.selectbox("Intent Label", INTENTS, index=INTENTS.index(default_intent) if default_intent in INTENTS else 0)
    decision = col2.selectbox("Expected Decision", DECISIONS, index=DECISIONS.index(default_dec) if default_dec in DECISIONS else 0)
    difficulty = col3.selectbox("Difficulty Tier", DIFFICULTIES, index=DIFFICULTIES.index(default_diff) if default_diff in DIFFICULTIES else 0)

    notes = st.text_input("Annotation Notes / Human Rationale", value=str(default_notes))

    submitted = st.form_submit_button("💾 Save Annotation")

    if submitted:
        new_row = {
            "id": case_id,
            "text": text,
            "intent": intent,
            "expected_decision": decision,
            "difficulty": difficulty,
            "notes": notes
        }
        # Update or append
        if is_already_labeled:
            df_golden.loc[df_golden["id"].astype(str) == case_id] = new_row
        else:
            df_golden = pd.concat([df_golden, pd.DataFrame([new_row])], ignore_index=True)

        os.makedirs(os.path.dirname(GOLDEN_CSV), exist_ok=True)
        df_golden.to_csv(GOLDEN_CSV, index=False, encoding="utf-8")
        st.success(f"Successfully saved Case #{idx} (`{case_id}`)!")
        st.rerun()

st.divider()
st.subheader("Current Golden Set Distribution")
if len(df_golden) > 0:
    col_d1, col_d2 = st.columns(2)
    col_d1.write("**Intent Breakdown:**")
    col_d1.dataframe(df_golden["intent"].value_counts())
    col_d2.write("**Difficulty Breakdown:**")
    col_d2.dataframe(df_golden["difficulty"].value_counts())
    st.write("**Dataset Table:**")
    st.dataframe(df_golden.tail(10))
