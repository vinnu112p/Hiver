# Hiver AI Customer Support System

A reproducible, evidence-grounded customer support pipeline evaluated on real-world Twitter support conversations (`SpotifyCares`). 

The system classifies incoming support requests into empirical intents, retrieves historically resolved precedents via dense vector search, synthesizes replies anchored strictly in past agent responses, and enforces a multi-signal safety escalation policy (`AUTO_HANDLE` vs `ESCALATE`).

---

## Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Headline Results](#headline-results)
- [Intent Taxonomy](#intent-taxonomy)
- [Sample Execution Traces](#sample-execution-traces)
- [Quickstart: Reproducing Results](#quickstart-reproducing-results)
- [Human-Judge Calibration](#human-judge-calibration)
- [Safety and Escalation Policy](#safety-and-escalation-policy)
- [Failure Analysis and Limitations](#failure-analysis-and-limitations)
- [Engineering Decisions](#engineering-decisions)
- [Repository Structure](#repository-structure)
- [Documentation Index](#documentation-index)

---

## Overview

Automating customer support in consumer applications requires strict safety boundaries. A support bot that hallucinates policies, invents refund promises, or fails to escalate account compromises causes customer churn and financial liability.

This project implements:
1. **Calibrated Intent Classification**: An n-gram TF-IDF linear classifier with class weighting that maps incoming queries into 7 empirical intents (**93.5% Accuracy**, **0.9245 Macro F1**) with sub-millisecond latency.
2. **Dense Precedent Grounding**: Queries retrieve top-3 historical resolutions from a **33,017-conversation knowledge base** using `sentence-transformers/all-MiniLM-L6-v2` and FAISS vector indexing.
3. **Evidence-Anchored Synthesis**: Drafts replies strictly from retrieved human-agent precedents, achieving **5.00/5.00 unsupported-claim safety** (zero hallucinated policies or financial commitments).
4. **Asymmetric Escalation**: Multi-signal decision engine evaluating classifier confidence, retrieval distance, sensitive financial dispute keywords, credential compromise, and explicit human agent requests. Optimizes for **Escalation Recall (91.1%)** over raw automation volume (43.0%), bounding **unsafe auto-handling to 5.0%**.

---

## System Architecture

```mermaid
flowchart TD
    subgraph STAGE1 ["Stage 1: Ingestion and Intent Classification"]
        A["Incoming Customer Query"] --> B["Normalization and Mention Cleaning (src.preprocess)"]
        B --> C["Calibrated N-Gram TF-IDF + Logistic Regression"]
        C --> D{"Confidence >= 0.35?"}
        D -->|Yes| E["Predicted Intent and Probability"]
        D -->|No| F["Structured JSON LLM Fallback (or Low-Confidence Tag)"]
        F --> E
    end

    subgraph STAGE2 ["Stage 2: Intent-Guided Semantic Retrieval"]
        E --> G["MiniLM-L6-v2 Dense Query Embedding (384-d)"]
        G --> H["FAISS Vector Search (10,000 Historical Vectors)"]
        H --> I["Intent-Prioritized Reranking (Top 3 Precedents)"]
    end

    subgraph STAGE3 ["Stage 3: Evidence-Grounded Synthesis"]
        I --> J["Evidence Guard (Top Similarity >= 0.35?)"]
        J -->|Yes| K["Grounded Generator (Anchored in Precedent Evidence)"]
        J -->|No| L["Conservative Escalation Fallback Template"]
    end

    subgraph STAGE4 ["Stage 4: Multi-Signal Safety Escalation"]
        E --> M["Multi-Signal Escalation Engine"]
        I --> M
        K --> M
        M --> N{"Safety Rules Passed?"}
        N -->|Security / Dispute / Low Conf / Human Req| O["ACTION: ESCALATE (With Stated Reason)"]
        N -->|Low Risk and Verified Precedent| P["ACTION: AUTO_HANDLE"]
    end
```

---

## Headline Results

Evaluated on the **200-case Golden Evaluation Benchmark** (`data/golden/golden_set.csv`) drawn from strictly newer conversations (`eval_pool.csv`):

| Evaluation Category | Metric | Majority Baseline | Simple ML (TF-IDF) | Final Grounded Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Classification** | **Accuracy** | 42.50% | 93.50% | **93.50%** |
| | **Macro F1** | 0.0852 | 0.9245 | **0.9245** |
| | **Weighted F1** | 0.2536 | 0.9347 | **0.9347** |
| | **Inference Latency** | < 0.1 ms | 0.4 ms | **~35 ms (End-to-End)** |
| **Response Quality (1–5)** | **Relevance** | — | — | **4.25** / 5.0 |
| | **Groundedness** | — | — | **4.99** / 5.0 |
| | **Helpfulness** | — | — | **4.25** / 5.0 |
| | **Brand Style** | — | — | **5.00** / 5.0 |
| | **Unsupported-Claim Safety** | — | — | **5.00** / 5.0 (Zero hallucinations) |
| **Safety and Escalation** | **Automation Rate** | — | — | **43.0%** (86/200 automated) |
| | **Escalation Recall** | — | — | **91.1%** (Caught 102/112 risks) |
| | **Escalation Precision** | — | — | **89.5%** (12 false alarms) |
| | **Unsafe Auto-Handling Rate** | — | — | **5.0%** (Strictly bounded) |

### Confusion Matrix (TF-IDF Baseline on Golden Benchmark)
![Confusion Matrix](evaluation/confusion_matrix.png)

---

## Intent Taxonomy

The 7 intents were derived empirically via frequency and n-gram clustering across 41,272 customer messages:

| Intent | Dataset Share | Risk Tier | Default Policy | Core Scope |
| :--- | :--- | :--- | :--- | :--- |
| `billing_subscription` | 19.0% | Medium | `auto_handle` | Charges, invoices, payment cards, Student/Family discount tiers |
| `cancellation_refund` | 0.7% | High | `escalate` | Terminating subscriptions, stopping recurring billing, refund claims |
| `account_access_security` | 10.7% | Critical | `escalate` | Locked accounts, password resets, compromised/hacked profiles |
| `audio_playback_issue` | 23.8% | Low | `auto_handle` | Stuttering, skipping tracks, offline download bugs, Bluetooth streaming |
| `app_crash_technical` | 7.8% | Low | `auto_handle` | App force-closes, black screens, freezing, update installation errors |
| `library_playlist_content` | 4.3% | Low | `auto_handle` | Missing tracks, playlist recovery, local files sync |
| `other_support` | 33.7% | Medium | `escalate` | Out-of-scope inquiries, community ideas, compliments, vague inputs |

*Detailed criteria, boundary definitions, and negative examples are documented in [INTENT_TAXONOMY.md](INTENT_TAXONOMY.md).*

---

## Sample Execution Traces

### Case 1: Billing Dispute -> Safe Escalation
```json
{
  "message": "I was charged twice for premium this month, please refund me!",
  "intent": {
    "label": "billing_subscription",
    "confidence": 0.7735,
    "source": "tfidf_primary"
  },
  "retrieved_cases": [
    {
      "case_id": "case_2285106_2285105",
      "similarity": 0.8376,
      "customer": "I was charged for premium but do not have premium on my account",
      "agent": "We've just sent a DM your way. Let's carry on chatting there /CG"
    }
  ],
  "reply": "We've just sent a DM your way. Let's carry on chatting there /CG",
  "decision": "escalate",
  "reason": "Disputed transaction or unauthorized charge requires billing team investigation.",
  "risk_level": "high",
  "evidence_strength": 0.8376,
  "grounding_summary": "Grounded in verified historical resolution (case case_2285106_2285105, similarity: 0.84)"
}
```

### Case 2: Routine Playback Issue -> Safe Auto-Handle
```json
{
  "message": "Every song stops playing after 10 seconds on my iPhone",
  "intent": {
    "label": "audio_playback_issue",
    "confidence": 0.4451,
    "source": "tfidf_primary"
  },
  "retrieved_cases": [
    {
      "case_id": "case_1970193_1970192",
      "similarity": 0.7189,
      "customer": "iphone 7 ios 11 or something. Whenever I try to skip songs my music just stops working.",
      "agent": "Hmm. Can you try logging out, restarting your device by holding Sleep/Wake + Volume Down, then logging in? /PB"
    }
  ],
  "reply": "Hmm. Can you try logging out, restarting your device by holding Sleep/Wake + Volume Down, then logging in? /PB",
  "decision": "auto_handle",
  "reason": "High intent confidence, strong historical precedent, and low risk.",
  "risk_level": "low",
  "evidence_strength": 0.7189,
  "grounding_summary": "Grounded in verified historical resolution (case case_1970193_1970192, similarity: 0.72)"
}
```

---

## Quickstart: Reproducing Results

The repository includes pre-built vector indexes and the 200-case Golden Evaluation Set. All core functionality runs locally on CPU without requiring external API keys or large dataset downloads.

### 1. Environment Setup
```bash
git clone https://github.com/vinnu112p/Hiver.git
cd Hiver

# Create and activate virtual environment (Python 3.10+)
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Evaluation (~7 seconds)
```bash
python -m src.evaluate
```
Evaluates the 200 Golden Set cases and writes summary metrics to `evaluation/results.md` and `evaluation/results.json`.

### 3. Run Unit Tests (19 passing)
```bash
python -m pytest tests/
```

### 4. Run Interactive Demo UI
```bash
streamlit run app/app.py
```
Opens a local Streamlit interface at `http://localhost:8501` to test custom inputs and inspect retrieved historical evidence.

### 5. Run Golden Benchmark Annotation Tool
```bash
streamlit run app/labeler.py
```
Opens the annotation dashboard to audit, filter, or review labels and human notes for all 200 evaluation benchmark cases.

### 6. CLI Inference
```bash
# Single message test
python -m src.pipeline --message "I was charged twice for premium"

# Batch evaluation on sample queries
python -m src.pipeline --input examples/sample_queries.json
```

---

## Human-Judge Calibration

To verify whether the automated rubric judge correlates with human assessment, we calibrated judge scores against human ratings across 30 diverse test cases (`evaluation/human_judge.csv`):

- **Within-1 Tolerance Agreement**: **100.0%**
- **Exact Score Agreement**: **70.0%**
- **Mean Absolute Error (MAE)**: **0.30**
- **Spearman Rank Correlation (rho)**: **0.5160** ($p = 0.0035 < 0.01$)

The statistically significant positive rank correlation and 100% within-1 agreement indicate that the automated judge is a reliable proxy for evaluating reply quality.

---

## Safety and Escalation Policy

The escalation engine enforces asymmetric risk handling: **False auto-handling is treated as a severe failure, while unnecessary escalation is treated as a minor efficiency cost.**

```mermaid
graph TD
    Trigger{"Query Risk Signals"} --> C1["Legal Threats or Scams"]
    Trigger --> C2["Explicit Human Request"]
    Trigger --> C3["Account Security / Password Reset"]
    Trigger --> C4["Financial Dispute / Unauthorized Charge"]
    Trigger --> C5["Out of Taxonomy (other_support)"]
    Trigger --> C6["Classifier Confidence < 0.35"]
    Trigger --> C7["Retrieval Similarity < 0.45"]

    C1 -->|Yes| ESC["ACTION: ESCALATE (High/Critical Risk)"]
    C2 -->|Yes| ESC
    C3 -->|Yes| ESC
    C4 -->|Yes| ESC
    C5 -->|Yes| ESC
    C6 -->|Yes| ESC
    C7 -->|Yes| ESC

    Trigger --> C8["All Checks Passed + Verified Precedent"]
    C8 --> AUTO["ACTION: AUTO_HANDLE (Low Risk)"]
```

---

## Failure Analysis and Limitations

### Top 5 Empirical Failure Modes
1. **Entangled Multi-Intent Inquiries**: User requests cancellation while blocked by a deleted Facebook SSO login (`case_662930_662929`). The system auto-handled cancellation guidance without resolving the credential blocker.
2. **Household Hardware Ambiguity**: Mentions of Bluetooth speakers for children fragmented classification probabilities between playback and Family Plan subscription (`case_560992_560991`).
3. **Feature Requests Misclassified as Bugs**: Constructive community suggestions sharing technical terms (`android`, `queue`, `play`) triggered low-confidence playback classification (`case_492392_492391`).
4. **Transport Network Errors Masked as Login Errors**: Offline device states trigger conservative login security escalation rules (`case_631985_631984`).
5. **Cross-Account Asset Transfers**: Moving playlists between unlinked accounts spans multiple subsystems without an atomic intent category (`case_617479_617478`).

*Detailed failure traces, hypotheses, and proposed improvements are documented in [evaluation/failure_analysis.md](evaluation/failure_analysis.md).*

### What Is Misleading About Our Headline Number?
*(From Section 11 of [REPORT.md](REPORT.md))*
- **Intent Accuracy != Resolution Safety**: A model can predict `billing_subscription` with 95% confidence and still draft an ungrounded or risky response. Real-world safety depends on the escalation engine and evidence verification.
- **Curated Golden Sets Mask Live Traffic Drift**: Real production traffic experiences abrupt shifts during outages, billing system upgrades, or app updates.
- **Historical Twitter Data Has Survivorship Bias**: Only captures users who chose to publicly tweet at support; silent drop-offs and users resolved via in-app help are omitted.
- **Static Knowledge Bases Contain Policy Drift**: Historical 2017 tweets may reference outdated URLs or deprecated UI settings without temporal validity filters.

---

## Engineering Decisions

Key decisions documented in [DECISION_LOG.md](DECISION_LOG.md):
- **Decision 1: Single Brand Focus (`SpotifyCares`)**: Eliminated cross-brand noise and PII leakage.
- **Decision 3: Preserving Negations**: Preserved tokens like `can't` and `not` because negation flips intent entirely.
- **Decision 5: Strict Temporal Splitting**: Partitioned data strictly by timestamp (older 80% to knowledge base; newer 20% to evaluation pool) to prevent future data leakage.
- **Decision 7: TF-IDF Primary with Hybrid Fallback**: Delivered 93.5% accuracy with sub-millisecond latency and zero API cost.
- **Decision 11: Asymmetric Escalation Priority**: Optimized Escalation Recall (91.1%) over raw automation volume (43.0%) to minimize unsafe auto-handling.

---

## Repository Structure

```
Hiver/
|-- app/
|   |-- app.py                     # Streamlit Interactive Demo UI
|   `-- labeler.py                 # Streamlit Golden Benchmark Annotation Dashboard
|-- data/
|   |-- golden/
|   |   |-- golden_set.csv         # 200 Curated Golden Evaluation Cases
|   |   `-- README.md              # Golden set sampling and label documentation
|   |-- sample/
|   |   `-- sample_queries.json    # Offline sample test queries
|   `-- README.md                  # Data source, columns, and licensing
|-- evaluation/
|   |-- baseline_majority.json     # Baseline 1 evaluation results
|   |-- baseline_tfidf.json        # Baseline 2 evaluation results
|   |-- brand_comparison.md        # Multi-brand comparative analysis
|   |-- confusion_matrix.png       # Baseline 2 Confusion Matrix visualization
|   |-- data_profile.md            # Profiling of 2.81M raw tweets
|   |-- escalation_results.json    # Escalation recall and safety metrics
|   |-- failure_analysis.md        # Top 5 empirical failure modes and hypotheses
|   |-- human_judge.csv            # 30-case human vs judge calibration dataset
|   |-- judge_agreement.json       # Human-judge statistical alignment metrics
|   `-- results.md                 # Complete headline evaluation report
|-- examples/
|   `-- sample_queries.json        # CLI query examples
|-- scripts/
|   |-- download_data.py           # Automated Kaggle/mirror dataset downloader
|   |-- curate_golden_set.py       # Golden set curation script
|   |-- prepare_data.ps1 / .sh     # End-to-end data preparation pipelines
|   |-- run_demo.ps1 / .sh         # Demo launcher scripts
|   `-- run_evaluation.ps1 / .sh   # Full evaluation pipeline scripts
|-- src/
|   |-- baseline_majority.py       # Baseline 1 (Majority class)
|   |-- baseline_tfidf.py          # Baseline 2 (TF-IDF + Logistic Regression)
|   |-- classifier.py              # Final confidence-gated intent classifier
|   |-- discover_intents.py        # Empirical intent discovery engine
|   |-- escalation.py              # Multi-signal safety escalation engine
|   |-- evaluate.py                # End-to-end evaluation harness
|   |-- generator.py               # Grounded reply synthesizer
|   |-- intents.py                 # Intent definitions and risk mappings
|   |-- judge.py                   # 5-dimension rubric judge
|   |-- pipeline.py                # Main orchestrator pipeline (CLI / API)
|   |-- preprocess.py              # Text cleaning and quality filters
|   |-- reconstruct_conversations.py # Two-pass streaming conversation builder
|   |-- retriever.py               # MiniLM-L6-v2 + FAISS vector retriever
|   `-- split_data.py              # Chronological train/eval partitioner
|-- tests/                         # PyTest suite (19 unit tests, all passing)
|-- DECISION_LOG.md                # 14 engineering decisions and trade-offs
|-- FINAL_CHECKLIST.md             # Quality assurance verification checklist
|-- INTENT_TAXONOMY.md             # Detailed taxonomy specifications
|-- PROJECT_COMPLETION.md          # 10 interview questions and project summary
|-- REPORT.md                      # Comprehensive technical report
|-- requirements.txt               # Dependency specifications
`-- README.md                      # Primary project guide
```

---

## Documentation Index

- **Technical Report**: [`REPORT.md`](REPORT.md)
- **Engineering Decisions**: [`DECISION_LOG.md`](DECISION_LOG.md)
- **Intent Taxonomy**: [`INTENT_TAXONOMY.md`](INTENT_TAXONOMY.md)
- **Golden Benchmark Data**: [`data/golden/golden_set.csv`](data/golden/golden_set.csv)
- **Failure Analysis**: [`evaluation/failure_analysis.md`](evaluation/failure_analysis.md)
- **Interview Preparation Guide**: [`PROJECT_COMPLETION.md`](PROJECT_COMPLETION.md)
- **Judge Agreement Metrics**: [`evaluation/judge_agreement.json`](evaluation/judge_agreement.json)
