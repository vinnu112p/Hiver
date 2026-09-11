"""
Phase 9: Baseline 1 — Trivial Majority-Class Baseline
Always predicts the most frequent class observed in the training distribution.
Evaluates on the golden set and computes:
- Accuracy
- Macro / Weighted F1
- Per-class Precision, Recall, F1
Saves results to evaluation/baseline_majority.json.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

GOLDEN_SET_PATH = os.path.join("data", "golden", "golden_set.csv")
TRAIN_DATA_PATH = os.path.join("data", "processed", "knowledge_base.csv")
OUT_JSON = os.path.join("evaluation", "baseline_majority.json")

def run_majority_baseline(golden_path=GOLDEN_SET_PATH, train_path=TRAIN_DATA_PATH):
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden set not found at {golden_path}.")

    df_golden = pd.read_csv(golden_path)
    y_true = df_golden['intent'].astype(str).tolist()

    # Determine majority class from train set if available, otherwise from golden set
    if os.path.exists(train_path) and 'intent' in pd.read_csv(train_path, nrows=10).columns:
        df_train = pd.read_csv(train_path)
        majority_class = df_train['intent'].value_counts().index[0]
    else:
        majority_class = pd.Series(y_true).value_counts().index[0]

    y_pred = [majority_class] * len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results = {
        "model": "Baseline 1 (Majority Class)",
        "predicted_constant_class": majority_class,
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(precision_macro, 4),
        "macro_recall": round(recall_macro, 4),
        "per_class_metrics": {
            cls_name: {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1-score"], 4),
                "support": metrics["support"]
            }
            for cls_name, metrics in report.items()
            if cls_name not in ["accuracy", "macro avg", "weighted avg"]
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Majority Baseline Evaluated on Golden Set ({len(y_true)} samples):")
    print(f"    - Constant Class: {majority_class}")
    print(f"    - Accuracy: {acc*100:.2f}%")
    print(f"    - Macro F1: {macro_f1:.4f}")
    print(f"[OK] Saved to: {OUT_JSON}")
    return results

if __name__ == "__main__":
    run_majority_baseline()
