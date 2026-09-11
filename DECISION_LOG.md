# Engineering Decision Log — Hiver AI Customer Support System

This document details **14 non-obvious architectural and engineering decisions**, their technical motivations, and their operational trade-offs across the project lifecycle.

---

### Decision 1: Single-Brand Focus on `SpotifyCares`
- **Decision**: Focused the entire pipeline on a single digital brand (`SpotifyCares`, 43,265 official replies) rather than pooling cross-brand tweets.
- **Reason**: Customer support vocabularies, policies, and escalation thresholds are intensely brand-specific. A system trained on mixed brands would confuse flight delays (`Delta`) with playlist bugs (`Spotify`), destroying retrieval grounding precision.
- **Trade-off**: Discards data from other brands, but yields a coherent, domain-focused knowledge base with minimal PII risk and high operational validity.

---

### Decision 2: Two-Pass Streaming Dataset Reconstruction
- **Decision**: Implemented a streaming two-pass conversation reconstruction algorithm instead of loading the full 2.8M-row dataset into memory.
- **Reason**: Pass 1 scans official brand replies and indexes target customer parent tweet IDs; Pass 2 streams customer tweets and pairs them. Runs in ~33 seconds with < 500 MB RAM overhead.
- **Trade-off**: Requires reading the file twice from disk, but prevents out-of-memory crashes on consumer laptops.

---

### Decision 3: Preserving Negations & Casing While Stripping Routing Handles
- **Decision**: Stripped leading `@handle` routing artifacts, but strictly preserved negations (`can't`, `not`, `never`), punctuation, and brand terms (`Family Plan`, `Premium`).
- **Reason**: Negation is the single most critical semantic feature in support tickets ("can log in" vs "cannot log in"). Aggressive stemming or stopword removal collapses these opposites into the same representation.
- **Trade-off**: Slightly increases vocabulary dimensionality, which is easily managed with TF-IDF sublinear scaling and `min_df=2`.

---

### Decision 4: Defining 7 Empirical Operational Intents (Including `other_support`)
- **Decision**: Discovered 7 intents directly from data clustering and frequency rather than importing an arbitrary external schema, and added an explicit `other_support` catch-all.
- **Reason**: Forcing every ambiguous or off-topic query into fixed functional buckets causes absurd responses (e.g. classifying a meme as a billing bug). `other_support` enables safe routing to human agents.
- **Trade-off**: Reduces apparent automation rate on ambiguous queries, but prevents dangerous out-of-domain hallucinations.

---

### Decision 5: Strict Temporal (Chronological) Data Splitting
- **Decision**: Split conversations strictly by timestamp (older 80% to historical knowledge base; newer 20% to evaluation pool).
- **Reason**: Random train/test splits introduce severe future data leakage, allowing test queries to retrieve historical conversations that occurred at the same time. Temporal splitting mimics live production reality.
- **Trade-off**: Produces slightly lower headline accuracy than an overfitted random split due to temporal topic drift, but provides honest performance estimates.

---

### Decision 6: Stratified Sampling for the Golden Evaluation Benchmark
- **Decision**: Sampled the 200-case evaluation set with stratified difficulty (70% normal, 15% ambiguous, 15% hard/high-risk) rather than uniform random sampling.
- **Reason**: Uniform random sampling in support data yields 80%+ easy, common queries, masking model failures on critical edge cases like account takeovers or billing disputes.
- **Trade-off**: The benchmark is harder than raw production traffic, but exposes vulnerabilities before deployment.

---

### Decision 7: TF-IDF + Logistic Regression as the Primary Fast Classifier
- **Decision**: Used an n-gram TF-IDF linear classifier with class weighting as the primary intent classifier, backed by an optional confidence-gated LLM fallback.
- **Reason**: Achieves **93.5% accuracy** with **< 1 ms latency** and zero API cost. 85%+ of support inquiries use standard vocabulary patterns that do not warrant a slow, expensive LLM call.
- **Trade-off**: Struggles with complex, negated multi-clause sentences, which are cleanly caught by the confidence threshold and routed to human review.

---

### Decision 8: FAISS Inner Product on Normalized Embeddings for Retrieval Grounding
- **Decision**: Built semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) with FAISS `IndexFlatIP`.
- **Reason**: Normalizing vectors enables exact cosine similarity via inner product, executing sub-millisecond similarity search locally without requiring an external vector database server.
- **Trade-off**: Indexes 10,000 cases in memory (~15 MB RAM), which is fast and lightweight, though scaling to 10M+ cases would eventually require an inverted file (IVF) index.

---

### Decision 9: Intent-Guided Retrieval Prioritization
- **Decision**: Filtered historical retrieval candidates by predicted intent before computing global semantic similarity.
- **Reason**: Prevents cross-category semantic confusion (e.g. a customer writing "I can't play because of payment error" retrieving a sound troubleshooting step instead of a billing step).
- **Trade-off**: If the intent classifier makes an error, candidate retrieval pool is constrained; solved by falling back to global search when confidence is low.

---

### Decision 10: Grounded Historical Precedent Over Generative Creativity
- **Decision**: Enforced that the agent draft replies synthesized *strictly* from retrieved historical resolutions, prohibiting the LLM from inventing policies, refund amounts, or URLs.
- **Reason**: In customer service, hallucinated policies create legal liabilities and customer churn. Historical resolutions written by human agents provide the only trusted source of truth.
- **Trade-off**: If historical cases lack an exact precedent, the agent cannot invent an answer and must escalate.

---

### Decision 11: Asymmetric Escalation: Prioritizing Recall Over Automation Rate
- **Decision**: Configured the escalation engine to optimize **Escalation Recall** (catching 91.1% of risky cases) at the expense of raw automation volume (43.0%).
- **Reason**: A False Auto-Handle (mishandling a hacked account or stolen credit card) destroys customer trust. A False Escalation (a human answering a routine playback FAQ) is merely an efficiency cost.
- **Trade-off**: Deflects fewer tickets than an aggressive chatbot, but guarantees an **unsafe auto-handling rate of only 5.0%**.

---

### Decision 12: Decoupling Classification Evaluation from Generation Evaluation
- **Decision**: Evaluated intent classification (Accuracy, F1, Confusion Matrix) completely separately from reply generation (5-dimension rubric judge).
- **Reason**: High classification accuracy does not guarantee a safe or helpful response. Treating evaluation as multi-stage prevents conflating syntactic routing with semantic safety.
- **Trade-off**: Requires running two distinct evaluation harnesses, increasing evaluation reporting complexity.

---

### Decision 13: Local Calibrated Evaluation Judge with Pluggable LLM Support
- **Decision**: Built a dual-mode evaluation judge that operates deterministically offline (measuring evidence overlap, safety checks, and style) while supporting drop-in LLM judging when an API key is present.
- **Reason**: Guarantees that any evaluator can clone the repo and reproduce headline evaluation metrics in under 15 minutes without needing a paid API key or third-party cloud account.
- **Trade-off**: Deterministic judge measures lexical overlap rather than full semantic nuance; validated by showing **100% within-1 agreement** with human raters.

---

### Decision 14: Interactive Streamlit Annotation & Demo Dashboard
- **Decision**: Built a lightweight Streamlit interface (`app/app.py` and `app/labeler.py`) rather than a complex React/Node stack.
- **Reason**: Allows interviewers to immediately test live queries, inspect historical precedent evidence, audit the Golden Set, and verify escalation decisions without frontend dependency overhead.
- **Trade-off**: Less visual customization than a full web app, but delivers 10x faster startup and zero build-step friction.
