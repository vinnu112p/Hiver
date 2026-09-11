#!/usr/bin/env bash
set -e

echo "=== HIVER AI SUPPORT AGENT: DATA PREPARATION PIPELINE ==="

echo "Step 1: Checking/Downloading Customer Support on Twitter..."
python scripts/download_data.py

echo "Step 2: Exploring dataset profile..."
python src/explore_data.py

echo "Step 3: Reconstructing SpotifyCares conversations..."
python -m src.reconstruct_conversations --brand SpotifyCares

echo "Step 4: Discovering empirical intent taxonomy..."
python -m src.discover_intents

echo "Step 5: Performing temporal train/evaluation split..."
python -m src.split_data

echo "Step 6: Sampling and curating golden benchmark..."
python scripts/sample_golden_candidates.py
python scripts/curate_golden_set.py

echo "[OK] Data preparation successfully finished!"
