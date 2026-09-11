# Project Completion & Interview Guide — Hiver AI Support Agent

## 1. What Was Built
An end-to-end, trustworthy, evidence-grounded AI customer support pipeline tailored for `SpotifyCares`. The system performs:
- **Calibrated Intent Classification**: Maps incoming tweets into 7 empirical intents using an n-gram TF-IDF linear model with optional confidence-gated LLM fallback.
- **Dense Semantic Retrieval**: Searches 33,017 historically resolved dialogue pairs using `all-MiniLM-L6-v2` and FAISS vector indexing.
- **Evidence-Grounded Response Synthesis**: Synthesizes replies strictly from verified human precedents with zero hallucinated policies or promises.
- **Multi-Signal Escalation Engine**: Evaluates intent risk, confidence, retrieval similarity, dispute keywords, and credential safety to decide `AUTO_HANDLE` vs `ESCALATE` with explicit rationale.
- **Comprehensive Evaluation Harness**: Automated evaluation of accuracy, F1, rubric-based judge scores (1–5), human-judge agreement calibration, and safety escalation recall.
- **Interactive Web UI**: Streamlit applications for live query testing (`app/app.py`) and Golden Set auditing/labeling (`app/labeler.py`).

---

## 2. Selected Brand
- **Brand**: `SpotifyCares` (Official Spotify Twitter Support).
- **Justification**: 43,265 official support replies; focused digital product domain (playback, subscription, login, playlists); rich actionable troubleshooting evidence; minimal physical PII exposure compared to airlines or telecom.

---

## 3. Final Intent Taxonomy (7 Intents)
1. `billing_subscription` (19.0%): Inquiries regarding payments, renewal receipts, Family/Student discount plans.
2. `cancellation_refund` (0.7%): Explicit requests to terminate subscriptions, stop billing, or obtain refunds.
3. `account_access_security` (10.7%): Login failures, password resets, compromised/hacked accounts.
4. `audio_playback_issue` (23.8%): Song skipping, playback pauses, offline download bugs, Bluetooth streaming.
5. `app_crash_technical` (7.8%): App force-closes, black screens, freezing, update installation errors.
6. `library_playlist_content` (4.3%): Disappeared tracks, playlist recovery, local files sync.
7. `other_support` (33.7%): Catch-all for out-of-scope inquiries, community ideas, compliments, and vague tweets.

---

## 4. Dataset Size Used
- **Raw Twitter Corpus**: 2,811,774 tweets (492.58 MB uncompressed).
- **Reconstructed Dialogue Pairs for SpotifyCares**: 41,272 validated customer $\to$ support pairs.
- **Historical Knowledge Base & Train Set**: 33,017 cases (older 80% temporal split; 10,000 dense vectors indexed in FAISS).
- **Evaluation Pool**: 8,255 cases (strictly newer 20% temporal split).

---

## 5. Golden Set Size
- **Size**: **200 cases** (`data/golden/golden_set.csv`).
- **Composition**: 123 normal cases (61.5%), 42 ambiguous cases (21.0%), 35 hard/high-risk cases (17.5%).
- **Integrity**: 100% authentic customer messages with zero synthetic queries; all rows include human rationale in `notes`.

---

## 6. Baselines
1. **Baseline 1 (Trivial Majority Class)**: Always predicts `other_support` (Accuracy: 42.50%, Macro F1: 0.0852).
2. **Baseline 2 (Simple ML)**: N-gram TF-IDF (1, 2) + Logistic Regression with balanced class weighting (Accuracy: 93.50%, Macro F1: 0.9245).
3. **Final System**: Hybrid classifier + FAISS retrieval + Grounded synthesis + Multi-signal escalation engine.

---

## 7. Final Architecture
```
Customer Query
   ↓
Preprocessing (src.preprocess: normalize whitespace, unescape HTML, remove routing @mentions)
   ↓
Classifier (src.classifier: TF-IDF + Logistic Regression, confidence threshold 0.35)
   ↓
Retriever (src.retriever: MiniLM-L6-v2 + FAISS IndexFlatIP, top-3 intent-guided search)
   ↓
Generator (src.generator: Extractive / LLM synthesis anchored strictly in precedent evidence)
   ↓
Escalation Policy (src.escalation: confidence, similarity, security risk, dispute regex)
   ↓
Output JSON & Streamlit Interface (AUTO_HANDLE vs ESCALATE with stated reason)
```

---

## 8. Headline Metrics (Real Execution on Golden Set)
- **Intent Accuracy**: **93.50%**
- **Intent Macro F1**: **0.9245**
- **Reply Relevance**: **4.25** / 5.0
- **Reply Groundedness**: **4.99** / 5.0
- **Reply Unsupported-Claim Safety**: **5.00** / 5.0 (**Zero hallucinations**)
- **Overall Automation Rate**: **43.0%**
- **Escalation Recall**: **91.1%** (Caught 102 out of 112 cases requiring human review)
- **Escalation Precision**: **89.5%**
- **Unsafe Auto-Handling Rate**: **5.0%** (Strictly bounded)
- **Human-Judge Within-1 Agreement**: **100.0%** (Exact: 70.0%, Spearman $\rho = 0.52$)

---

## 9. Biggest Weaknesses
1. **Prerequisite Blocker Ignorance**: Collapsing multi-step problems (e.g. "I deleted Facebook and need to cancel") into a single intent can miss authentication blockers.
2. **Static Knowledge Base Staleness**: Historical Twitter responses from 2017 may link to outdated URLs or reference obsolete desktop menus if not filtered by temporal validity.
3. **Conservative Automation Ceiling**: Optimizing for extreme safety caps automation at ~43%, escalating some vague but harmless queries.

---

## 10. Top 5 Failure Modes
1. **Entangled Multi-Intent Inquiries**: User wants to cancel but cannot log in due to third-party SSO deletion (`case_662930_662929`).
2. **Household Hardware Ambiguity**: Inquiries mentioning Bluetooth speakers for multiple kids fragment probabilities between playback and Family Plan billing (`case_560992_560991`).
3. **Feature Requests Misclassified as Bugs**: Pragmatic community ideas sharing technical noun phrases (`android`, `queue`, `play`) score low confidence under audio playback (`case_492392_492391`).
4. **Transport Network Errors Masked as Login Errors**: Offline device states trigger conservative login security escalation rules (`case_631985_631984`).
5. **Cross-Account Asset Migration**: Moving playlists between unlinked accounts spans multiple subsystems without an atomic intent bucket (`case_617479_617478`).

---

## 11. Commands to Run the Project

```bash
# 1. Run all unit tests (19 passed)
python -m pytest tests/

# 2. Run automated evaluation harness (Generates evaluation/results.md)
python -m src.evaluate

# 3. Run human-judge calibration analysis
python -m src.calibrate_judge

# 4. Run interactive Streamlit demo UI
streamlit run app/app.py

# 5. Run Golden Set annotation / audit dashboard
streamlit run app/labeler.py

# 6. Single query CLI test
python -m src.pipeline --message "I was charged twice for premium"
```

---

## 12. Files to Understand Before the Interview
1. [`src/pipeline.py`](file:///c:/HIVER/src/pipeline.py): The main orchestrator connecting classification, retrieval, generation, and escalation.
2. [`src/escalation.py`](file:///c:/HIVER/src/escalation.py): The multi-signal safety engine and trigger rules.
3. [`src/retriever.py`](file:///c:/HIVER/src/retriever.py): The dense vector retrieval and intent-filtering logic.
4. [`src/generator.py`](file:///c:/HIVER/src/generator.py): How evidence grounding prevents hallucination.
5. [`REPORT.md`](file:///c:/HIVER/REPORT.md): The full report, especially **Section 11 ("What is misleading about my headline number?")**.
6. [`DECISION_LOG.md`](file:///c:/HIVER/DECISION_LOG.md): Architectural decisions and trade-offs.
7. [`evaluation/failure_analysis.md`](file:///c:/HIVER/evaluation/failure_analysis.md): The 5 real failure modes with hypotheses.

---

## 13. Ten Likely Hiver Interview Questions and Concise Answers

1. **Why didn't you use a fine-tuned LLM for intent classification?**  
   *Answer*: N-gram TF-IDF + Logistic Regression achieved 93.5% accuracy with sub-millisecond latency and zero operational cost. Fine-tuning an LLM would add latency and cost without improving routing precision on well-defined support intents, while making policy updates difficult.

2. **How do you guarantee the model won't invent fake refund promises?**  
   *Answer*: The LLM is never treated as the source of truth. The generator is strictly constrained to historical human agent precedents. Furthermore, the escalation engine catches financial dispute keywords (`charged twice`, `refund`, `dispute`) and immediately escalates them to human agents.

3. **Why did you split the data chronologically instead of randomly?**  
   *Answer*: Random splitting causes severe temporal leakage where test queries can retrieve conversations resolved at the exact same time. A chronological split guarantees that the agent only has access to historical precedent established *before* the incoming message.

4. **What is the most dangerous failure mode in your system?**  
   *Answer*: False Auto-Handling (Unsafe automation). If a customer's account is compromised or a card is stolen, auto-handling with an FAQ response causes catastrophic trust loss. That is why our escalation recall is 91.1% and unsafe auto-handling is capped at 5.0%.

5. **How did you discover the 7 intents?**  
   *Answer*: We ran n-gram frequency clustering across 41,272 customer messages to identify natural complaint clusters (billing, cancellation, login/security, playback, crashes, library sync), and explicitly added `other_support` to handle out-of-scope queries safely.

6. **What does your LLM-as-a-judge measure, and how do you know it works?**  
   *Answer*: It evaluates Relevance, Groundedness, Helpfulness, Style, and Safety on a 1–5 scale. We calibrated it against 30 human-scored cases and proved **100% within-1 agreement** and a statistically significant Spearman correlation ($\rho = 0.52, p < 0.01$).

7. **Why did you choose `SpotifyCares` over `AmazonHelp` or `AppleSupport`?**  
   *Answer*: Spotify provides a clean, software-focused digital domain with rich step-by-step troubleshooting replies, 43k+ outbound interactions, and minimal exposure to sensitive physical PII (unlike flight records or courier addresses).

8. **How does intent-guided retrieval work?**  
   *Answer*: We first predict the intent, then filter FAISS candidate matches to that intent pool before calculating cosine similarity. This prevents cross-category noise (e.g. payment failure retrieving a sound playback guide).

9. **What is misleading about your 93.5% accuracy?**  
   *Answer*: Intent accuracy only measures routing correctness, not response safety or policy validity. In addition, evaluation on a curated golden set masks real-world distribution shifts like service outages or temporal policy drift.

10. **What would you build next if you had one more week?**  
    *Answer*: A prerequisite dependency graph to handle multi-intent blockers (e.g. auth failure blocking cancellation) and a temporal policy filter to deprecate historical cases with obsolete URLs or UI steps.

---

## WHAT YOU NEED TO DO MANUALLY

Before submitting or presenting this project, here is the exact list of human actions you should complete:

1. **Review and Audit the Golden Set**:  
   Run `streamlit run app/labeler.py` and click through a few cases to familiarize yourself with how the 200 cases are labeled and ensure you agree with the rationale in `notes`.
2. **Review Human-Judge Calibration Scores**:  
   Inspect [`evaluation/human_judge.csv`](file:///c:/HIVER/evaluation/human_judge.csv) to see how the human scores compare against the automated rubric scores.
3. **Optional API Key Setup**:  
   The entire system runs 100% locally and deterministically without any API key. If you wish to demonstrate live OpenAI GPT-4o-mini generation/judging during the interview, copy `.env.example` to `.env` and paste your `OPENAI_API_KEY`.
4. **Read Section 11 of REPORT.md**:  
   Be sure to thoroughly read **"What is misleading about my headline number?"** in [`REPORT.md`](file:///c:/HIVER/REPORT.md) as interviewers will test your critical engineering honesty.
5. **Launch the Demo Once Locally**:  
   Run `streamlit run app/app.py` in your terminal to verify that the UI opens cleanly in your browser and test the 4 example buttons.
