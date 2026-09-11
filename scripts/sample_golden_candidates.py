"""
Phase 8: Golden Set Stratified Sampler & Curation Script
Extracts a stratified candidate pool of exactly 200 evaluation cases from data/processed/eval_pool.csv.
Follows the mandatory sampling strategy:
- 70% normal/common support inquiries (140 cases across intents)
- 15% ambiguous / multi-intent / vague cases (30 cases)
- 15% difficult / high-risk / security / dispute cases (30 cases)
Saves candidate pool to data/golden/candidates_to_label.csv.
"""
import os
import re
import pandas as pd

EVAL_POOL_PATH = os.path.join("data", "processed", "eval_pool.csv")
CANDIDATES_PATH = os.path.join("data", "golden", "candidates_to_label.csv")
GOLDEN_CSV_PATH = os.path.join("data", "golden", "golden_set.csv")

RE_SECURITY = re.compile(r"\b(login|password|hacked|stolen|compromised|reset|locked out|access)\b", re.IGNORECASE)
RE_BILLING_DISPUTE = re.compile(r"\b(charged twice|double charge|unauthorized|refund|cancel|money back|dispute)\b", re.IGNORECASE)
RE_TECHNICAL = re.compile(r"\b(crash|freeze|update|black screen|bug|error)\b", re.IGNORECASE)
RE_PLAYBACK = re.compile(r"\b(play|stops|skipping|offline|sound|bluetooth|headphones)\b", re.IGNORECASE)
RE_LIBRARY = re.compile(r"\b(playlist|library|album|local files|disappeared|deleted|greyed out)\b", re.IGNORECASE)
RE_AMBIGUOUS = re.compile(r"\b(why|what|help|broken|not working|fix this|issue|problem)\b", re.IGNORECASE)

def sample_candidates():
    if not os.path.exists(EVAL_POOL_PATH):
        raise FileNotFoundError(f"Evaluation pool not found at {EVAL_POOL_PATH}")

    df = pd.read_csv(EVAL_POOL_PATH)
    print(f"[*] Sifting candidate pool from {len(df):,} strictly newer evaluation cases...")

    # Bucket cases by characteristics
    sec_cases = df[df['customer_message'].str.contains(RE_SECURITY, na=False)]
    disp_cases = df[df['customer_message'].str.contains(RE_BILLING_DISPUTE, na=False)]
    tech_cases = df[df['customer_message'].str.contains(RE_TECHNICAL, na=False)]
    play_cases = df[df['customer_message'].str.contains(RE_PLAYBACK, na=False)]
    lib_cases = df[df['customer_message'].str.contains(RE_LIBRARY, na=False)]
    
    # Ambiguous / Short / Vague
    short_vague = df[(df['customer_message'].str.split().str.len() <= 6) & df['customer_message'].str.contains(RE_AMBIGUOUS, na=False)]

    sampled = []
    seen_ids = set()

    def add_sample(sub_df, count, target_diff, category_tag):
        added = 0
        for _, row in sub_df.sample(frac=1.0, random_state=42).iterrows():
            cid = str(row['case_id'])
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            sampled.append({
                "id": cid,
                "text": str(row['customer_message']),
                "category_tag": category_tag,
                "target_difficulty": target_diff,
                "support_response_reference": str(row.get('support_response', ''))
            })
            added += 1
            if added >= count:
                break

    # 1. High-risk & Difficult cases (30 total)
    add_sample(sec_cases, 15, "hard", "security_risk")
    add_sample(disp_cases, 15, "hard", "financial_dispute")

    # 2. Ambiguous & Vague cases (30 total)
    add_sample(short_vague, 20, "ambiguous", "short_vague")
    # Multi-signal cases
    multi_signal = df[df['customer_message'].str.contains(RE_PLAYBACK, na=False) & df['customer_message'].str.contains(RE_TECHNICAL, na=False)]
    add_sample(multi_signal, 10, "ambiguous", "multi_intent")

    # 3. Normal / Standard functional support cases (140 total)
    add_sample(play_cases, 40, "normal", "playback")
    add_sample(tech_cases, 30, "normal", "crash_technical")
    add_sample(lib_cases, 25, "normal", "playlist_library")
    add_sample(disp_cases, 25, "normal", "standard_billing")
    add_sample(df, 20, "normal", "general_support")

    df_cand = pd.DataFrame(sampled[:200])
    os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
    df_cand.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8")
    print(f"[OK] Sampled {len(df_cand)} stratified evaluation candidates -> {CANDIDATES_PATH}")
    return df_cand

if __name__ == "__main__":
    sample_candidates()
