#!/usr/bin/env bash
set -e

echo "=== HIVER AI SUPPORT AGENT: EVALUATION HARNESS ==="

echo "1. Running Baseline 1 (Majority Class)..."
python -m src.baseline_majority

echo "2. Running Baseline 2 (TF-IDF + Logistic Regression)..."
python -m src.baseline_tfidf

echo "3. Running Full End-to-End System Evaluation (Golden Benchmark)..."
python -m src.evaluate

echo "4. Running Human-Judge Calibration Analysis..."
python -m src.calibrate_judge

echo "5. Executing Unit Test Suite..."
python -m pytest tests/

echo "[OK] Evaluation completed! Review results in evaluation/results.md"
