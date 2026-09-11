"""
Phase 7: Temporal Data Splitting
Partitions processed support cases chronologically into:
1. Retrieval Knowledge Base & Training Set (older 80%)
2. Evaluation Pool (strictly newer 20%)
This prevents data leakage: no evaluation query can retrieve its own future resolution.
"""
import os
import argparse
import pandas as pd

PROCESSED_FILE = os.path.join("data", "processed", "support_cases.csv")
RETRIEVAL_FILE = os.path.join("data", "processed", "knowledge_base.csv")
EVAL_POOL_FILE = os.path.join("data", "processed", "eval_pool.csv")

def split_data_temporally(input_file=PROCESSED_FILE, split_ratio=0.80):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed cases not found at {input_file}. Run reconstruct_conversations.py first.")

    print(f"[*] Loading processed support cases from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"    Total cases available: {len(df):,}")

    # Parse timestamps for strict temporal split
    # Raw created_at example: "Tue Oct 31 22:10:47 +0000 2017"
    print("    Parsing timestamps for temporal ordering...")
    df['parsed_time'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    
    # Sort chronologically (oldest first)
    df_sorted = df.sort_values(by='parsed_time', ascending=True).reset_index(drop=True)

    # Calculate split point
    n_train = int(len(df_sorted) * split_ratio)
    df_train = df_sorted.iloc[:n_train].drop(columns=['parsed_time'])
    df_eval = df_sorted.iloc[n_train:].drop(columns=['parsed_time'])

    os.makedirs(os.path.dirname(RETRIEVAL_FILE), exist_ok=True)
    df_train.to_csv(RETRIEVAL_FILE, index=False, encoding="utf-8")
    df_eval.to_csv(EVAL_POOL_FILE, index=False, encoding="utf-8")

    print(f"[OK] Temporal split completed successfully:")
    print(f"    - Historical Knowledge Base & Train Set: {len(df_train):,} cases ({split_ratio*100:.0f}%) -> {RETRIEVAL_FILE}")
    print(f"    - Evaluation Pool (strictly newer cases): {len(df_eval):,} cases ({(1-split_ratio)*100:.0f}%) -> {EVAL_POOL_FILE}")

    return df_train, df_eval

if __name__ == "__main__":
    split_data_temporally()
