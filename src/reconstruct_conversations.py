"""
Phase 4: Conversation Reconstruction
Reconstructs grounded customer-support conversation pairs (customer -> official brand reply)
for a selected brand (default: SpotifyCares).
Ensures:
- Temporal order & parent-child validation
- Brand verification (outbound reply must come from the official handle)
- Noise rejection & deduplication via src.preprocess
- Clean output saved to data/processed/support_cases.csv
"""
import os
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_customer_text, clean_support_text, is_valid_interaction

RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
OUT_DATA_PATH = os.path.join("data", "processed", "support_cases.csv")

def reconstruct_pairs(raw_file=RAW_DATA_PATH, out_file=OUT_DATA_PATH, brand="SpotifyCares", chunk_size=250000):
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw dataset not found at {raw_file}. Please run scripts/download_data.py first.")

    print(f"[*] Starting conversation reconstruction for brand: '{brand}'")
    start_time = time.time()
    
    # PASS 1: Identify all official replies from the target brand that reply to a parent tweet
    print("    Pass 1/2: Scanning brand replies and collecting parent customer tweet IDs...")
    brand_replies_by_parent = {} # parent_id -> list of reply records
    brand_total_replies = 0

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size, low_memory=False, dtype=str):
        brand_mask = (chunk['author_id'] == brand) & (chunk['in_response_to_tweet_id'].notna())
        matches = chunk[brand_mask]
        brand_total_replies += len(matches)

        for _, row in matches.iterrows():
            parent_id = str(row['in_response_to_tweet_id']).strip().split(".")[0]
            if not parent_id or parent_id == 'nan':
                continue
            
            # Store the brand reply
            reply_record = {
                "support_tweet_id": str(row['tweet_id']).strip().split(".")[0],
                "support_author_id": brand,
                "support_raw_text": str(row['text']),
                "support_created_at": str(row['created_at'])
            }
            if parent_id not in brand_replies_by_parent:
                brand_replies_by_parent[parent_id] = []
            brand_replies_by_parent[parent_id].append(reply_record)

    needed_parents = set(brand_replies_by_parent.keys())
    print(f"    Pass 1 complete: Found {brand_total_replies:,} total replies from {brand}.")
    print(f"    Targeting {len(needed_parents):,} unique parent customer tweet IDs.")

    # PASS 2: Scan for matching parent customer tweets
    print("    Pass 2/2: Finding parent customer messages and pairing...")
    paired_cases = []
    seen_pairs = set()

    for chunk in pd.read_csv(raw_file, chunksize=chunk_size, low_memory=False, dtype=str):
        # We only care about inbound tweets whose tweet_id is in needed_parents
        chunk['clean_tweet_id'] = chunk['tweet_id'].astype(str).str.strip().str.split('.').str[0]
        inbound_mask = chunk['inbound'].astype(str).str.lower().isin(['true', '1'])
        matched_chunk = chunk[inbound_mask & chunk['clean_tweet_id'].isin(needed_parents)]

        for _, cust_row in matched_chunk.iterrows():
            cust_id = cust_row['clean_tweet_id']
            cust_author = str(cust_row['author_id'])
            cust_raw_text = str(cust_row['text'])
            cust_created = str(cust_row['created_at'])

            # Pair with all recorded brand responses to this parent
            for reply in brand_replies_by_parent[cust_id]:
                supp_id = reply["support_tweet_id"]
                supp_raw = reply["support_raw_text"]

                # Quality gatekeeper from src.preprocess
                is_valid, reason = is_valid_interaction(cust_raw_text, supp_raw)
                if not is_valid:
                    continue

                cust_clean = clean_customer_text(cust_raw_text)
                supp_clean = clean_support_text(supp_raw)

                pair_key = (cust_clean, supp_clean)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                paired_cases.append({
                    "case_id": f"case_{cust_id}_{supp_id}",
                    "brand": brand,
                    "customer_tweet_id": cust_id,
                    "customer_author_id": cust_author,
                    "customer_message": cust_clean,
                    "customer_raw_text": cust_raw_text,
                    "support_tweet_id": supp_id,
                    "support_response": supp_clean,
                    "support_raw_text": supp_raw,
                    "created_at": cust_created,
                    "support_created_at": reply["support_created_at"]
                })

    df_paired = pd.DataFrame(paired_cases)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_paired.to_csv(out_file, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"[OK] Reconstructed {len(df_paired):,} high-quality customer-support pairs in {elapsed:.1f}s")
    print(f"[OK] Saved to: {out_file}")
    return df_paired

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct customer-support conversation pairs.")
    parser.add_argument("--brand", type=str, default="SpotifyCares", help="Brand support handle to extract")
    args = parser.parse_args()
    reconstruct_pairs(brand=args.brand)
