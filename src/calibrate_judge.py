"""
Phase 18: Human-Judge Calibration & Agreement Harness
Measures empirical alignment between human scoring and the automated rubric judge
across a 30-case calibration sample.
Computes:
- Exact Agreement Rate
- Within-1 Integer Tolerance Rate (Standard operational agreement benchmark)
- Spearman Rank Correlation (rho)
- Mean Absolute Error (MAE)
Saves:
- evaluation/human_judge.csv
- evaluation/judge_agreement.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HUMAN_JUDGE_CSV = os.path.join("evaluation", "human_judge.csv")
AGREEMENT_JSON = os.path.join("evaluation", "judge_agreement.json")
RESULTS_JSON = os.path.join("evaluation", "results.json")
GOLDEN_CSV = os.path.join("data", "golden", "golden_set.csv")

def evaluate_judge_agreement(csv_path=HUMAN_JUDGE_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Human judge dataset not found at {csv_path}. Run generate_calibration_sample() first.")

    df = pd.read_csv(csv_path)
    print(f"[*] Calculating alignment statistics across {len(df)} human-judged cases...")

    human_scores = df["human_score"].astype(float).values
    judge_scores = df["judge_score"].astype(float).values

    # 1. Exact Agreement
    exact_matches = np.sum(human_scores == judge_scores)
    exact_rate = float(exact_matches / len(df))

    # 2. Within-1 Tolerance Agreement (|human - judge| <= 1)
    diffs = np.abs(human_scores - judge_scores)
    within_1_matches = np.sum(diffs <= 1.0)
    within_1_rate = float(within_1_matches / len(df))

    # 3. Mean Absolute Error (MAE)
    mae = float(np.mean(diffs))

    # 4. Spearman Rank Correlation
    if len(set(human_scores)) > 1 and len(set(judge_scores)) > 1:
        corr, p_value = spearmanr(human_scores, judge_scores)
    else:
        corr, p_value = 0.0, 1.0

    agreement_results = {
        "sample_size": len(df),
        "exact_agreement_rate": round(exact_rate, 4),
        "within_1_tolerance_rate": round(within_1_rate, 4),
        "mean_absolute_error": round(mae, 4),
        "spearman_correlation": round(float(corr), 4),
        "p_value": round(float(p_value), 6),
        "summary": (
            f"The judge exhibits {within_1_rate*100:.1f}% within-1 agreement with human raters "
            f"(Exact: {exact_rate*100:.1f}%, MAE: {mae:.2f}, Spearman rho: {corr:.2f})."
        )
    }

    os.makedirs(os.path.dirname(AGREEMENT_JSON), exist_ok=True)
    with open(AGREEMENT_JSON, "w", encoding="utf-8") as f:
        json.dump(agreement_results, f, indent=2)

    print(f"[OK] Judge Agreement Evaluated:")
    print(f"    - Within-1 Agreement: {within_1_rate*100:.1f}%")
    print(f"    - Exact Agreement: {exact_rate*100:.1f}%")
    print(f"    - MAE: {mae:.2f}")
    print(f"    - Spearman rho: {corr:.4f} (p={p_value:.4e})")
    print(f"[OK] Saved to: {AGREEMENT_JSON}")

    return agreement_results

if __name__ == "__main__":
    evaluate_judge_agreement()
