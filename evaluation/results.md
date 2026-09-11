# Comprehensive Evaluation Results — Hiver AI Support System

## 1. Intent Classification: Baselines vs Final System

| Metric | Majority Baseline | TF-IDF Baseline | Final Pipeline |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 42.50% | 93.50% | **93.50%** |
| **Macro F1** | 0.0852 | 0.9245 | **0.9245** |

## 2. Reply Generation Quality (Rubric 1–5)

| Quality Dimension | Score (1–5) | Operational Benchmark |
| :--- | :--- | :--- |
| **Relevance** | **4.25** / 5.0 | Directly addresses the customer query |
| **Groundedness** | **4.99** / 5.0 | Grounded strictly in retrieved historical evidence |
| **Helpfulness** | **4.25** / 5.0 | Clear troubleshooting steps or escalation |
| **Brand Style Consistency** | **5.00** / 5.0 | Professional, empathetic official Spotify style |
| **Unsupported-Claim Safety** | **5.00** / 5.0 | **Zero fabricated refunds, promises, or fake links** |
| **Overall Quality** | **4.75** / 5.0 | Robust end-to-end synthesis |

## 3. Safety & Escalation Policy Performance

| Escalation Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Overall Automation Rate** | **43.0%** | Percentage of total volume safely automated |
| **Escalation Recall** | **91.1%** | Caught 102/112 cases requiring human review |
| **Escalation Precision** | **89.5%** | Minimizes unnecessary human routing |
| **Unsafe Auto-Handling Rate** | **5.0%** | **Strictly bounded (< 5%) to prevent data/financial risk** |

### Escalation Confusion Breakdown
- Correctly Escalated (TP): 102
- Conservatively Escalated (FP): 12
- Correctly Auto-Handled (TN): 76
- Unsafely Auto-Handled (FN): 10