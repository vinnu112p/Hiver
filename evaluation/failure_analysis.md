# Empirical Failure Analysis — Top 5 Failure Modes

This document analyzes **real failures** observed when running the end-to-end support pipeline against the 200-case Golden Evaluation Benchmark. 

No simulated or hypothetical failures are included; all examples, confidence scores, and retrieval outputs are drawn directly from execution logs.

---

## Failure Mode 1: Entangled Multi-Intent Inquiries (Auth Dependency on Third-Party SSO)

### Real Example:
> **Customer Query (`case_662930_662929`):**  
> *"I permanently deleted my Facebook which is linked to my Spotify account. I want to cancel my subscription but can't find a way to log in without using my facebook details. Help"*

- **Expected Result**:  
  - Intent: `cancellation_refund` or `account_access_security`  
  - Decision: `ESCALATE` (Customer cannot authenticate due to deleted 3rd-party credentials; bot cannot cancel an account the user cannot access).
- **Actual Pipeline Result**:  
  - Predicted Intent: `cancellation_refund` (Confidence: `0.7412`)  
  - Decision: `AUTO_HANDLE`  
  - Grounded Reply: Auto-generated guidance on canceling subscriptions via web settings.
- **Why It Failed**:  
  The linear classifier matched strong cancellation keywords (`cancel my subscription`), giving high confidence (> 0.70). However, the underlying blocker was credential authentication (`permanently deleted my Facebook`). Because the escalation rule for cancellation checked for explicit financial dispute tokens (`charged twice`, `refund`, `fraud`) rather than credential loss, the system incorrectly assumed the standard cancellation self-service flow was viable.
- **Hypothesis**:  
  Single-label intent classification collapses multi-step customer journeys into a single dominant verb. When a customer has a pre-condition failure (cannot log in) preceding an objective (cancel account), single-intent models miss the prerequisite blocker.
- **Concrete Improvement**:  
  Implement a multi-intent detector or a prerequisite intent dependency check. If an authentication failure signal co-occurs with a billing/cancellation request, elevate the risk tier to `critical` and trigger mandatory human escalation.

---

## Failure Mode 2: Complex Household Hardware & Multi-User Plan Inquiries

### Real Example:
> **Customer Query (`case_560992_560991`):**  
> *"hello! I have bought my 3 kids Bluetooth speakers for Xmas and would like us all to be able to use Spotify separately at the same time. Is this possible? I already have an account just for me."*

- **Expected Result**:  
  - Intent: `billing_subscription` (Family Plan inquiry)  
  - Decision: `AUTO_HANDLE` (Direct user to Spotify Premium for Family pricing & setup).
- **Actual Pipeline Result**:  
  - Predicted Intent: `audio_playback_issue` (Confidence: `0.3383`)  
  - Decision: `ESCALATE`  
  - Reason: `Low intent classification confidence (0.34 < 0.35)`.
- **Why It Failed**:  
  The message contains lexical signals from two opposing categories: hardware audio tokens (`Bluetooth speakers`, `use at the same time`) vs subscription plan tokens (`kids`, `separately`, `already have an account`). The TF-IDF probability was fragmented across `audio_playback_issue` (34%) and `billing_subscription` (28%), neither exceeding the 0.35 confidence threshold.
- **Hypothesis**:  
  Bag-of-words and n-gram models cannot distinguish semantic focus (the core question is about subscription account sharing) from conversational context (the speakers are just Christmas gifts).
- **Concrete Improvement**:  
  For low-confidence queries exhibiting bimodal probability distributions between two specific categories, invoke the structured LLM fallback prompt to disambiguate the primary user goal before triggering escalation.

---

## Failure Mode 3: Product Feedback & UX Feature Suggestions Misclassified as Technical Bugs

### Real Example:
> **Customer Query (`case_492392_492391`):**  
> *"You know what would be really useful? On android, being able to add a button to my home screen (Like you can add a route w/ google maps) that's basically "play whatever is next in queue". It's currently 3-4 button presses, and annoying UX."*

- **Expected Result**:  
  - Intent: `other_support` (Community feature request / feedback)  
  - Decision: `ESCALATE` (or auto-route to Community Ideas board).
- **Actual Pipeline Result**:  
  - Predicted Intent: `audio_playback_issue` (Confidence: `0.2538`)  
  - Decision: `ESCALATE`  
  - Reason: `Low intent classification confidence (0.25 < 0.35)`.
- **Why It Failed**:  
  While the decision was correctly escalated due to low confidence, the intent classification failed because words like `android`, `play`, `queue`, and `button` strongly overlapped with audio playback and app bugs.
- **Hypothesis**:  
  Customer complaints and feature suggestions share identical noun-phrase vocabularies (`android`, `queue`, `play`). The difference lies entirely in communicative pragmatic framing (`"You know what would be really useful?"`).
- **Concrete Improvement**:  
  Add a dedicated semantic feature for discourse mood (interrogative / imperative vs subjunctive / feature proposal) or train a lightweight sentiment/intent gatekeeper that routes feature requests directly to `other_support` / community feedback.

---

## Failure Mode 4: Network Connectivity Bugs Masked as Account Login Failures

### Real Example:
> **Customer Query (`case_631985_631984`):**  
> *"I can't log in in my account. It's says that I'm offline but I'm certainly not aince i can tweet you and do other stuffs that requires wlan!!"*

- **Expected Result**:  
  - Intent: `app_crash_technical` or `audio_playback_issue` (App offline cache synchronization bug)  
  - Decision: `AUTO_HANDLE` (Provide standard steps: toggle airplane mode, verify Spotify background data permissions).
- **Actual Pipeline Result**:  
  - Predicted Intent: `account_access_security` (Confidence: `0.3255`)  
  - Decision: `ESCALATE`  
  - Reason: `Potential account security compromise or credential takeover requires human verification.`
- **Why It Failed**:  
  The customer wrote `"I can't log in in my account"`. Under our strict safety-first policy, any query containing login failure phrases defaults to `account_access_security` with mandatory human escalation.
- **Hypothesis**:  
  Conservative security rules prioritize safety over automation rate. The customer is not actually locked out of their credentials; Spotify's client cannot reach its auth servers over WLAN. However, false auto-handling a real lockout is catastrophic, whereas conservatively escalating a network bug is merely inefficient.
- **Concrete Improvement**:  
  Differentiate between *credential invalidity* ("wrong password", "email changed") and *transport-level auth failure* ("says I'm offline", "connection error on login").

---

## Failure Mode 5: Account Asset Migration Inquiries Across Unlinked Profiles

### Real Example:
> **Customer Query (`case_617479_617478`):**  
> *"Hello. I dont want to login via FB anymore, but create a separate Spotify account. Can you please help me with moving my music (lists and downloads) across. Thank you"*

- **Expected Result**:  
  - Intent: `library_playlist_content` or `account_access_security`  
  - Decision: `AUTO_HANDLE` (Provide official link to public playlist sharing / collaborative playlist duplication guide).
- **Actual Pipeline Result**:  
  - Predicted Intent: `other_support` (Confidence: `0.3990`)  
  - Decision: `ESCALATE`  
  - Reason: `Customer query falls outside defined automation taxonomy.`
- **Why It Failed**:  
  The customer's request is non-standard: moving saved playlists and downloads from an old Facebook-linked account to a newly created independent account. The vocabulary is diffuse (`login via FB`, `create separate account`, `moving my music`, `downloads across`).
- **Hypothesis**:  
  Long-tail administrative workflows that touch multiple subsystems (Authentication + Account Creation + Library Export) do not conform to atomic intent taxonomies.
- **Concrete Improvement**:  
  Incorporate cross-account migration procedures into the historical retrieval knowledge base with an intent tag of `library_playlist_content`. Even if classified as `other_support`, a high retrieval similarity score (> 0.80) to a verified account migration case could allow safe semi-automated resolution.
