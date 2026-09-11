"""
Phase 8: Interactive Streamlit Labeler Tool
Provides an intuitive, web-based UI for manual inspection, labeling, and auditing
of the 200 Golden Evaluation Set cases.
"""
import os
import sys
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hiver Golden Set Audit Tool", layout="wide")

GOLDEN_CSV = os.path.join("data", "golden", "golden_set.csv")

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

st.title("Hiver AI Support — Golden Set Annotation and Audit Tool")
st.markdown("Inspect, audit, and modify customer evaluation cases for ground-truth benchmark analysis.")

# Load dataset
if not os.path.exists(GOLDEN_CSV):
    st.error(f"Golden benchmark not found at {GOLDEN_CSV}.")
    st.stop()

df_golden = pd.read_csv(GOLDEN_CSV)

# Progress metrics
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Golden Cases", len(df_golden))
col_m2.metric("Auto-Handle Cases", len(df_golden[df_golden['expected_decision'] == 'auto_handle']))
col_m3.metric("Escalation Cases", len(df_golden[df_golden['expected_decision'] == 'escalate']))

st.divider()

# Select case index to inspect / annotate
idx = st.number_input("Case Index (0-199)", min_value=0, max_value=len(df_golden)-1, value=0)

case_row = df_golden.iloc[idx]
case_id = str(case_row["id"])
text = str(case_row["text"])
cur_intent = str(case_row.get("intent", INTENTS[0]))
cur_dec = str(case_row.get("expected_decision", "auto_handle"))
cur_diff = str(case_row.get("difficulty", "normal"))
cur_notes = str(case_row.get("notes", ""))

st.subheader(f"Case #{idx} — ID: `{case_id}`")
st.info(f"**Customer Message:**\n\n\"{text}\"")

with st.form("label_form"):
    col1, col2, col3 = st.columns(3)

    intent = col1.selectbox("Intent Label", INTENTS, index=INTENTS.index(cur_intent) if cur_intent in INTENTS else 0)
    decision = col2.selectbox("Expected Decision", DECISIONS, index=DECISIONS.index(cur_dec) if cur_dec in DECISIONS else 0)
    difficulty = col3.selectbox("Difficulty Tier", DIFFICULTIES, index=DIFFICULTIES.index(cur_diff) if cur_diff in DIFFICULTIES else 0)

    notes = st.text_input("Annotation Notes / Human Rationale", value=cur_notes)

    submitted = st.form_submit_button("Save Annotation")

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
