# Full System Evaluation Pipeline (PowerShell)
Write-Host "=== HIVER AI SUPPORT AGENT: EVALUATION HARNESS ===" -ForegroundColor Cyan

Write-Host "`n1. Running Baseline 1 (Majority Class)..." -ForegroundColor Yellow
python -m src.baseline_majority

Write-Host "`n2. Running Baseline 2 (TF-IDF + Logistic Regression)..." -ForegroundColor Yellow
python -m src.baseline_tfidf

Write-Host "`n3. Running Full End-to-End System Evaluation (Golden Benchmark)..." -ForegroundColor Yellow
python -m src.evaluate

Write-Host "`n4. Running Human-Judge Calibration Analysis..." -ForegroundColor Yellow
python -m src.calibrate_judge

Write-Host "`n5. Executing Unit Test Suite..." -ForegroundColor Yellow
python -m pytest tests/

Write-Host "`n[OK] Evaluation completed! Review results in evaluation/results.md" -ForegroundColor Green
