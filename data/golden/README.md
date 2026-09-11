# Golden Evaluation Benchmark (200 Cases)

## 1. Purpose & Guarantees

This golden dataset (`golden_set.csv`) serves as the ground-truth evaluation benchmark for the Hiver AI Customer Support System. 

**Core Integrity Rules:**
- **Zero Synthetic/Hallucinated Queries**: Every single text sample is an authentic customer tweet sent to Spotify support.
- **Zero Future Data Leakage**: All 200 examples are drawn strictly from the temporal evaluation pool (`data/processed/eval_pool.csv`), which contains only conversations occurring *after* the historical training and retrieval corpus cutoff.
- **Auditable Human Rationale**: Every row includes difficulty classification (`normal`, `ambiguous`, `hard`), expected escalation decision (`auto_handle` vs `escalate`), and explicit human rationale in the `notes` column.

---

## 2. Sampling Methodology: Why This Is NOT Simple Random Sampling

Simple random sampling of customer support queries results in severe representation collapse:
- The top 2 most common complaints dominate 70%+ of the test set.
- High-severity security issues (credential compromise, fraudulent charges) and rare cancellation disputes appear in under 1% of random draws.
- Vague or multi-intent edge cases are under-represented, creating an illusion of high model performance.

### Stratified Sampling Strategy:
1. **Normal / Common Cases (~60–70%)**: Clear, single-intent inquiries regarding playback bugs, clean app reinstallation, playlist synchronization, and standard billing questions.
2. **Ambiguous Cases (~15–20%)**: Under-specified customer messages ("why won't it work?", "help please"), multi-intent complaints (e.g. playback stuttering combined with app freeze), and messages lacking device or account context.
3. **Difficult / High-Risk Cases (~15–20%)**:
   - Compromised or hacked accounts (Russian listening history, unrecognized email changes).
   - Financial billing disputes (unauthorized charges, double billing).
   - Cancellation / refund demands.
   - Explicit customer requests demanding to speak with a human manager or agent.

---

## 3. Dataset Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique case identifier linking back to raw customer and support tweet IDs |
| `text` | String | Normalized customer message (mentions cleaned, negations preserved) |
| `intent` | String | Ground-truth intent label from the 7 empirical intents |
| `expected_decision` | String | `auto_handle` or `escalate` |
| `difficulty` | String | `normal`, `ambiguous`, or `hard` |
| `notes` | String | Human annotator justification for the label and decision |

---

## 4. Disagreement Handling & Labeling Tool

To ensure complete transparency and allow independent verification or modification of any label:
1. **CLI Label Tool**: Run `python -m src.label_tool` for interactive terminal-based inspection and labeling.
2. **Streamlit UI Labeler**: Run `streamlit run app/labeler.py` for a full visual dashboard allowing interactive browsing, filtering, and live editing of all 200 cases.
