# Brand Selection Comparison — Customer Support on Twitter

| Brand | Outbound Tweets | Domain | Intent Feasibility | Privacy/Noise Risk | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SpotifyCares` | 43,265 | Digital Music Streaming | High (clean software/billing domain: playback, login, billing, offline, playlist, subscription) | Low (mostly redirects to troubleshooting URLs or DM without raw phone/address leaks) | **Selected (Optimal size, focused software intents, clean grounded replies)** |
| `AppleSupport` | 106,860 | Hardware & OS Devices | Medium (hardware repairs, iOS version variations, high hardware dependency) | Low to Medium | **Viable alternative but high multi-device complexity** |
| `AmazonHelp` | 169,840 | E-Commerce & Logistics | Very High volume, but heavily tracking/carrier dependent | Medium (order numbers, delivery addresses) | **Very large, high volume of tracking inquiries** |
| `Uber_Support` | 56,270 | Rideshare & Food Delivery | Medium (driver disputes, fares, cancellations) | High (trip locations, pickup times) | **Heavy escalation rate due to physical incident reports** |
| `Delta` | 42,253 | Airlines & Travel | Medium (flight delays, rebooking, baggage) | High (PNR numbers, passport info) | **High regulatory and real-time flight status dependency** |
| `sprintcare` | 22,381 | Telecom Carrier | Medium (billing, sim cards, cell tower outages) | High (phone numbers, account pins) | **Generic DM deflection rate is very high** |

## Selected Brand: `SpotifyCares`

### Rationale:
1. **Domain Suitability**: Spotify support revolves around a focused digital product (audio playback, subscription tiers, login/credentials, app crashes, playlist sync, and billing disputes). This maps naturally to 6–8 well-defined, mutually exclusive intents.
2. **Data Scale**: `SpotifyCares` offers tens of thousands of high-quality interactions—large enough to train accurate classifiers and maintain a comprehensive historical retrieval store, yet compact enough to embed and index locally in minutes.
3. **Evidence Grounding Quality**: Spotify agents frequently provided structured, reusable troubleshooting steps (e.g. clean reinstall steps, offline sync clearing, payment verification) that provide rich grounding evidence for RAG response generation.
4. **Safety & PII**: Unlike airlines (`Delta`) or telecom (`sprintcare`), Spotify conversations have lower exposure to sensitive physical identifiers (flight PNRs, physical locations, SIM card PINs), making them safer for synthetic grounding demonstration.