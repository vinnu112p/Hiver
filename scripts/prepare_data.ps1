# Data Preparation Pipeline (PowerShell)
Write-Host "=== HIVER AI SUPPORT AGENT: DATA PREPARATION PIPELINE ===" -ForegroundColor Cyan

Write-Host "`nStep 1: Checking/Downloading Customer Support on Twitter..." -ForegroundColor Yellow
python scripts/download_data.py

Write-Host "`nStep 2: Exploring dataset profile..." -ForegroundColor Yellow
python src/explore_data.py

Write-Host "`nStep 3: Reconstructing SpotifyCares conversations..." -ForegroundColor Yellow
python -m src.reconstruct_conversations --brand SpotifyCares

Write-Host "`nStep 4: Discovering empirical intent taxonomy..." -ForegroundColor Yellow
python -m src.discover_intents

Write-Host "`nStep 5: Performing temporal train/evaluation split..." -ForegroundColor Yellow
python -m src.split_data

Write-Host "`nStep 6: Sampling and curating golden benchmark..." -ForegroundColor Yellow
python scripts/sample_golden_candidates.py
python scripts/curate_golden_set.py

Write-Host "`n[OK] Data preparation successfully finished!" -ForegroundColor Green
