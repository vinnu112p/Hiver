# Customer Support Intent Taxonomy — SpotifyCares

This taxonomy was empirically derived from frequency and n-gram analysis of **41,272 customer interactions** from the Twitter customer support dataset (`SpotifyCares`).

It consists of **7 distinct intents** designed to cover operational digital music streaming inquiries while maintaining clear escalation boundaries.

---

## 1. `billing_subscription`
- **Description**: Inquiries regarding payment methods, monthly billing cycles, subscription pricing, renewal receipts, and student/family plan eligibility.
- **Inclusion Criteria**: Queries asking about recurring charges, credit card updates, subscription tier differences, payment failures, or verification of student discounts.
- **Exclusion Criteria**: Demands for refunds or explicit cancellation of accounts (classified under `cancellation_refund`).
- **Real Examples from Data**:
  1. *"Why was my card charged $9.99 when I signed up for the student discount?"*
  2. *"How do I update my payment info for Spotify Premium?"*
  3. *"My family plan invite link says it expired, how do I resend it?"*
  4. *"Does premium automatically renew every month?"*
- **Ambiguous Example**: *"Spotify just charged me twice this month."* (Classified as `billing_subscription`, but escalates to human due to disputed transaction keywords).
- **Default Escalation Policy**: `AUTO_HANDLE` for standard FAQs; `ESCALATE` if disputed charge keywords are detected.

---

## 2. `cancellation_refund`
- **Description**: Explicit requests to terminate a subscription, close an account, or receive money back for unwanted or unauthorized renewal charges.
- **Inclusion Criteria**: Customer explicitly mentions "cancel", "cancelling", "refund", "unsubscribe", or "stop charging me".
- **Exclusion Criteria**: Questions about switching between tiers without cancelling (classified under `billing_subscription`).
- **Real Examples from Data**:
  1. *"I want to cancel my premium subscription immediately."*
  2. *"You charged me after I cancelled my free trial, I need a refund."*
  3. *"How do I unsubscribe from Spotify? The button in settings is missing."*
- **Ambiguous Example**: *"I'm going to Apple Music if you don't fix this bug."* (Classified under `app_crash_technical` or `other_support`, not cancellation, as there is no cancellation action requested).
- **Default Escalation Policy**: `ESCALATE` (Direct monetary refund execution requires authorized human verification).

---

## 3. `account_access_security`
- **Description**: Problems logging in, password reset failures, credential stuffing, unrecognized profile changes, or suspected compromised/hacked accounts.
- **Inclusion Criteria**: Inability to log in, password reset emails not arriving, notices of email change not initiated by the user, or unfamiliar songs in listening history.
- **Exclusion Criteria**: App crashes occurring before the login screen (classified under `app_crash_technical`).
- **Real Examples from Data**:
  1. *"I can't log into my account and the password reset email never arrives."*
  2. *"Someone changed the email on my Spotify account and I'm locked out."*
  3. *"My account has weird Russian songs in my recently played, I think I was hacked."*
  4. *"Forgot which email is linked to my Spotify username."*
- **Ambiguous Example**: *"It logged me out suddenly while playing music."* (Borderline with playback session limit; classified as `account_access_security` for safety).
- **Default Escalation Policy**: `ESCALATE` (Critical safety risk; automated agents must never disclose or manipulate credentials).

---

## 4. `audio_playback_issue`
- **Description**: Music stops unexpectedly, songs buffer or skip after a few seconds, downloaded songs fail to play offline, or audio output distortion occurs over Bluetooth/speakers.
- **Inclusion Criteria**: Audio playback interruptions, track skipping, offline cache errors, audio quality issues, or device-specific playback stops.
- **Exclusion Criteria**: The entire application freezes or terminates (classified under `app_crash_technical`).
- **Real Examples from Data**:
  1. *"Every song stops playing after 10 seconds on my iPhone."*
  2. *"My downloaded offline songs won't play when I have no service."*
  3. *"Songs keep pausing randomly when connected to Bluetooth in my car."*
  4. *"Spotify web player says audio playback is not supported on this browser."*
- **Ambiguous Example**: *"The app stops when I press play."* (If app force-closes, `app_crash_technical`; if playback pauses, `audio_playback_issue`).
- **Default Escalation Policy**: `AUTO_HANDLE` (Standard troubleshooting steps: clean reinstall, clearing local cache, toggling offline mode).

---

## 5. `app_crash_technical`
- **Description**: Client app force-closing, black/blank screen on launch, UI unresponsiveness, freezing during startup, or installation update failures.
- **Inclusion Criteria**: Mentions of crash, force close, freeze, black screen, update failure, or glitch on iOS, Android, macOS, or Windows.
- **Exclusion Criteria**: Playback stops while UI remains responsive (classified under `audio_playback_issue`).
- **Real Examples from Data**:
  1. *"Spotify desktop app keeps crashing immediately when I open it on Windows 10."*
  2. *"Latest iOS update broke the app, it just shows a black screen."*
  3. *"App freezes every time I click on search bar."*
  4. *"Cannot install Spotify on my Mac, gets stuck on installer."*
- **Ambiguous Example**: *"Your update ruined everything."* (Vague; if no specific error details, classified as `other_support`).
- **Default Escalation Policy**: `AUTO_HANDLE` (Provides verified official clean-reinstallation and OS cache-clearing procedures).

---

## 6. `library_playlist_content`
- **Description**: Inquiries regarding user playlists, disappeared saved tracks, local file synchronization from desktop to mobile, greyed-out unplayable tracks, or podcast catalog availability.
- **Inclusion Criteria**: Missing user playlists, songs disappearing from 'Liked Songs', local audio files not syncing across devices, or album metadata errors.
- **Exclusion Criteria**: The song is visible and playable but stutters or crashes audio engine (classified under `audio_playback_issue`).
- **Real Examples from Data**:
  1. *"All the songs in my workout playlist just disappeared."*
  2. *"Why are half the songs on this album greyed out and unplayable?"*
  3. *"How do I sync local mp3 files from my computer to my phone on Spotify?"*
  4. *"Can I recover a playlist I accidentally deleted last week?"*
- **Ambiguous Example**: *"Why can't I find Taylor Swift's new song?"* (Could be licensing or search; classified under `library_playlist_content`).
- **Default Escalation Policy**: `AUTO_HANDLE` (Guides user to playlist recovery tool in account settings, licensing explanations, or local files guide).

---

## 7. `other_support`
- **Description**: Fallback category for out-of-scope customer queries, marketing promotions, artist tagging, compliments, memes, or underspecified one-word tweets.
- **Inclusion Criteria**: Queries that do not match the operational scope of the 6 functional support intents.
- **Exclusion Criteria**: Any query that contains clear keywords matching the top 6 intents.
- **Real Examples from Data**:
  1. *"Hey Spotify check out my new single on your platform!"*
  2. *"When is Spotify Wrapped coming out this year?"*
  3. *"You guys have the best app ever thank you!"*
  4. *"Help me please"* (Vague, lacks actionable detail).
- **Default Escalation Policy**: `ESCALATE` (Requires human review to clarify user intent).
