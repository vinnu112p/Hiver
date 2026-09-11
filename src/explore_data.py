"""
Phase 2: Data Exploration Script
Inspects customer-support-on-twitter raw dataset in streaming chunks for memory efficiency.
Computes real dataset statistics and generates evaluation/data_profile.json and evaluation/data_profile.md.
"""
import os
import sys
import json
import time
from collections import Counter
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
OUT_JSON_PATH = os.path.join("evaluation", "data_profile.json")
OUT_MD_PATH = os.path.join("evaluation", "data_profile.md")

def explore_dataset(file_path=RAW_DATA_PATH, chunk_size=200000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset not found at {file_path}. Run scripts/download_data.py first.")

    print(f"[*] Beginning data profiling on: {file_path}")
    start_time = time.time()
    
    total_rows = 0
    inbound_counts = Counter()
    missing_counts = Counter()
    brand_outbound_counts = Counter()
    char_lengths = []
    has_response_id_count = 0
    has_in_response_to_count = 0
    
    # Process in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
        total_rows += len(chunk)
        
        # Missing values
        for col in chunk.columns:
            missing_counts[col] += int(chunk[col].isna().sum())
            
        # Inbound vs Outbound
        chunk['inbound_bool'] = chunk['inbound'].astype(str).str.lower().isin(['true', '1'])
        inbound_counts.update(chunk['inbound_bool'].value_counts().to_dict())
        
        # Support brands (outbound authors)
        outbound_authors = chunk[~chunk['inbound_bool']]['author_id'].astype(str)
        brand_outbound_counts.update(outbound_authors.value_counts().to_dict())
        
        # Response linkage counts
        has_resp = chunk['response_tweet_id'].notna() & (chunk['response_tweet_id'].astype(str).str.strip() != '')
        has_in_resp = chunk['in_response_to_tweet_id'].notna() & (chunk['in_response_to_tweet_id'].astype(str).str.strip() != '')
        has_response_id_count += int(has_resp.sum())
        has_in_response_to_count += int(has_in_resp.sum())
        
        # Sample character lengths (take 1000 per chunk to keep array size manageable)
        valid_texts = chunk['text'].dropna().astype(str)
        if len(valid_texts) > 0:
            sample_sub = valid_texts.sample(min(1000, len(valid_texts)), random_state=42)
            char_lengths.extend(sample_sub.str.len().tolist())
            
        print(f"    Processed {total_rows:,} rows... ({time.time() - start_time:.1f}s)", flush=True)

    elapsed = time.time() - start_time
    print(f"[OK] Completed streaming analysis of {total_rows:,} rows in {elapsed:.1f}s")
    
    # Summary stats
    char_lengths = np.array(char_lengths)
    stats = {
        "dataset_name": "Customer Support on Twitter (thoughtvector/customer-support-on-twitter)",
        "file_path": file_path,
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
        "total_rows": total_rows,
        "columns": list(missing_counts.keys()),
        "missing_values": dict(missing_counts),
        "inbound_distribution": {
            "inbound_customer": int(inbound_counts.get(True, 0)),
            "outbound_brand": int(inbound_counts.get(False, 0)),
            "inbound_percentage": round(inbound_counts.get(True, 0) / max(1, total_rows) * 100, 2)
        },
        "response_linkages": {
            "has_response_tweet_id": has_response_id_count,
            "has_in_response_to_tweet_id": has_in_response_to_count,
            "percentage_with_direct_parent": round(has_in_response_to_count / max(1, total_rows) * 100, 2)
        },
        "text_length_chars_sampled": {
            "mean": round(float(np.mean(char_lengths)), 2),
            "median": round(float(np.median(char_lengths)), 2),
            "std": round(float(np.std(char_lengths)), 2),
            "min": int(np.min(char_lengths)),
            "max": int(np.max(char_lengths))
        },
        "top_20_brand_support_accounts": [
            {"brand": brand, "outbound_replies": count}
            for brand, count in brand_outbound_counts.most_common(20)
        ]
    }
    
    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved JSON profile: {OUT_JSON_PATH}")
    
    # Generate Markdown Report
    md_lines = [
        "# Dataset Profile — Customer Support on Twitter",
        "",
        f"- **Total Rows**: {stats['total_rows']:,}",
        f"- **File Size**: {stats['file_size_mb']} MB",
        f"- **Inbound Customer Tweets**: {stats['inbound_distribution']['inbound_customer']:,} ({stats['inbound_distribution']['inbound_percentage']}%)",
        f"- **Outbound Support Replies**: {stats['inbound_distribution']['outbound_brand']:,}",
        f"- **Tweets with Direct Parent (`in_response_to_tweet_id`)**: {stats['response_linkages']['has_in_response_to_tweet_id']:,} ({stats['response_linkages']['percentage_with_direct_parent']}%)",
        "",
        "## Missing Values per Column",
        "",
        "| Column | Missing Count | Missing % |",
        "| :--- | :--- | :--- |",
    ]
    for col, count in stats["missing_values"].items():
        pct = round(count / stats["total_rows"] * 100, 2)
        md_lines.append(f"| `{col}` | {count:,} | {pct}% |")
        
    md_lines.extend([
        "",
        "## Sampled Message Length (Characters)",
        "",
        f"- **Mean**: {stats['text_length_chars_sampled']['mean']}",
        f"- **Median**: {stats['text_length_chars_sampled']['median']}",
        f"- **Std**: {stats['text_length_chars_sampled']['std']}",
        f"- **Min / Max**: {stats['text_length_chars_sampled']['min']} / {stats['text_length_chars_sampled']['max']}",
        "",
        "## Top 20 Brand Support Accounts (Outbound Volume)",
        "",
        "| Rank | Brand Support Handle | Outbound Replies |",
        "| :--- | :--- | :--- |"
    ])
    
    for rank, item in enumerate(stats["top_20_brand_support_accounts"], 1):
        md_lines.append(f"| {rank} | `{item['brand']}` | {item['outbound_replies']:,} |")
        
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Saved Markdown profile: {OUT_MD_PATH}")
    
    return stats

if __name__ == "__main__":
    explore_dataset()
