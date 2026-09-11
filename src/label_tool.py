"""
Phase 8: Golden Set Labeling Tool
Interactive CLI utility for manual annotation of evaluation cases.
Ensures zero fabricated human labels.
Allows the user to inspect, label, tag difficulty, and add rationale.
"""
import os
import sys
import pandas as pd

GOLDEN_CSV = os.path.join("data", "golden", "golden_set.csv")
EVAL_POOL_CSV = os.path.join("data", "processed", "eval_pool.csv")

INTENT_MAP = {
    "1": "billing_subscription",
    "2": "cancellation_refund",
    "3": "account_access_security",
    "4": "audio_playback_issue",
    "5": "app_crash_technical",
    "6": "library_playlist_content",
    "7": "other_support"
}

DECISION_MAP = {
    "1": "auto_handle",
    "2": "escalate"
}

DIFFICULTY_MAP = {
    "1": "normal",
    "2": "ambiguous",
    "3": "hard"
}

def run_label_tool():
    print("=" * 60)
    print("  HIVER GOLDEN EVALUATION SET — MANUAL LABELING TOOL")
    print("=" * 60)
    print("Commands:")
    print("  Intents: 1=billing, 2=cancellation, 3=account_security, 4=playback, 5=app_crash, 6=library, 7=other")
    print("  Decisions: 1=auto_handle, 2=escalate")
    print("  Difficulty: 1=normal, 2=ambiguous, 3=hard")
    print("  Type 'q' anytime to save and quit.")
    print("=" * 60)

    # Load existing or create new
    if os.path.exists(GOLDEN_CSV):
        df_golden = pd.read_csv(GOLDEN_CSV)
        labeled_ids = set(df_golden['id'].astype(str).tolist())
    else:
        df_golden = pd.DataFrame(columns=["id", "text", "intent", "expected_decision", "difficulty", "notes"])
        labeled_ids = set()

    if not os.path.exists(EVAL_POOL_CSV):
        print(f"Error: Evaluation pool not found at {EVAL_POOL_CSV}. Please run split_data.py first.")
        return

    df_eval = pd.read_csv(EVAL_POOL_CSV)
    unlabeled = df_eval[~df_eval['case_id'].astype(str).isin(labeled_ids)]
    print(f"[*] Already labeled: {len(labeled_ids)} | Remaining to label: {len(unlabeled)}")

    new_rows = []
    for _, row in unlabeled.iterrows():
        case_id = str(row['case_id'])
        text = str(row['customer_message'])
        supp = str(row.get('support_response', ''))

        print("\n" + "-" * 50)
        print(f"CASE ID: {case_id}")
        print(f"CUSTOMER TEXT:\n\"{text}\"")
        if supp:
            print(f"HISTORICAL BRAND REPLY:\n\"{supp[:120]}...\"")
        print("-" * 50)

        # Prompt intent
        choice_intent = input("Select Intent [1-7, or q to quit]: ").strip().lower()
        if choice_intent == 'q':
            break
        intent = INTENT_MAP.get(choice_intent, "other_support")

        # Prompt decision
        choice_dec = input("Select Decision [1=auto_handle, 2=escalate]: ").strip()
        decision = DECISION_MAP.get(choice_dec, "escalate")

        # Prompt difficulty
        choice_diff = input("Select Difficulty [1=normal, 2=ambiguous, 3=hard]: ").strip()
        diff = DIFFICULTY_MAP.get(choice_diff, "normal")

        # Prompt notes
        notes = input("Annotation Notes (optional): ").strip()

        record = {
            "id": case_id,
            "text": text,
            "intent": intent,
            "expected_decision": decision,
            "difficulty": diff,
            "notes": notes
        }
        new_rows.append(record)
        df_golden = pd.concat([df_golden, pd.DataFrame([record])], ignore_index=True)
        df_golden.to_csv(GOLDEN_CSV, index=False, encoding="utf-8")
        print(f"[OK] Saved case {case_id} (Total Golden Set: {len(df_golden)})")

    print(f"\n[OK] Session closed. Golden set now has {len(df_golden)} entries in {GOLDEN_CSV}")

if __name__ == "__main__":
    run_label_tool()
