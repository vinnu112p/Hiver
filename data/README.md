# Customer Support on Twitter — Data Documentation

## 1. Dataset Overview

- **Source**: [Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Verified Mirror**: [HuggingFace: SunidhiSriram/twcs](https://huggingface.co/datasets/SunidhiSriram/twcs)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Format**: CSV (~492.58 MB uncompressed, ~2,811,774 tweets)

## 2. Expected Columns

The raw dataset contains 7 columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `tweet_id` | Integer/String | Unique identifier for each tweet |
| `author_id` | String | Anonymized user ID (e.g. `115712`) or official brand handle (e.g. `SpotifyCares`, `AppleSupport`) |
| `inbound` | Boolean (`True`/`False`) | `True` if sent by a customer inbound to the brand; `False` if sent by the brand support handle |
| `created_at` | String / Timestamp | RFC 2822 / Twitter formatted timestamp (e.g. `Tue Oct 31 22:10:47 +0000 2017`) |
| `text` | String | The actual tweet text, preserving mentions, emojis, and punctuation |
| `response_tweet_id` | String | Comma-separated list of tweet IDs that responded to this tweet |
| `in_response_to_tweet_id` | String | The tweet ID to which this tweet is replying (null for initiating tweets) |

## 3. How to Obtain the Dataset

### Method A: Automated Download (Recommended)
Run the project download script:
```powershell
python scripts/download_data.py
```
This fetches `twcs.csv` into `data/raw/twcs.csv` with progress monitoring and integrity checks.

### Method B: Official Kaggle CLI
If you have Kaggle credentials configured in `~/.kaggle/kaggle.json`:
```powershell
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
```

## 4. Why the Full Dataset Is Not Committed to Git

1. **Size & Performance**: The raw CSV is ~493 MB and contains over 2.8 million rows. Storing it in Git violates repository size best practices and degrades clone speed.
2. **Reproducibility**: `data/raw/` is added to `.gitignore`. Instead, deterministic scripts construct:
   - `data/processed/support_cases.csv`: Cleaned, paired brand support interactions.
   - `data/sample/sample_queries.json`: Lightweight offline demonstration queries.
   - `data/golden/golden_set.csv`: Curated, human-labelled evaluation benchmark (150–250 cases).

## 5. Subsetting and Preprocessing

- **Brand-Focused Filtering**: Real-world customer support is brand-specific (support policies, technical vocabulary, and escalation criteria for music streaming are completely different from an airline or mobile carrier).
- **Conversation Threading**: Using `in_response_to_tweet_id` and `response_tweet_id`, we match initiating customer tweets (`inbound=True`) to official support replies (`inbound=False`).
- **Temporal Integrity**: Chronological sorting ensures that older interactions form the historical retrieval and training corpus, while strictly newer interactions form the evaluation testbed (no forward-looking data leakage).
