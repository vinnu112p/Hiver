# Project Status — Hiver AI Customer Support System

**Date**: 2026-09-11  
**Project**: Hiver SDE Intern Take-Home — Trustworthy, Grounded AI Support Agent  

---

## 1. Environment Discovery

- **Operating System**: Windows (AMD64 / PowerShell)
- **Python Version**: `3.13.0`
- **Package Manager**: `pip` (25.3)
- **Git**: Installed (`git version 2.47.0.windows.1`), initialized in `C:\HIVER`
- **Git User Config**: `vinnu112p` / `patelvinnu112@gmail.com`
- **Installed Key Libraries**:
  - `pandas` (2.3.0)
  - `numpy` (2.3.0)
  - `scikit-learn` (1.8.0)
  - `sentence-transformers` (5.2.0)
  - `torch` (2.9.1)
  - `transformers` (4.57.3)
  - `faiss-cpu` (1.13.2)
  - `streamlit` (1.63.0)
  - `openai` (1.6.1)
  - `tqdm`, `pydantic`
- **Internet Connectivity**: Active and verified
- **API Keys / Credentials**:
  - No external proprietary LLM API key pre-set in environment.
  - Architecture decision: System runs deterministically with TF-IDF + FAISS + lightweight embeddings locally, with optional pluggable LLM generation/judge (OpenAI/Anthropic/compatible endpoint) when an API key is provided, falling back to a deterministic grounded template/extractive generator if no external API key is present.

---

## 2. Dataset Availability

- **Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` / `twcs.csv`)
- **Local status**: Not present initially in workspace.
- **Acquisition Source Verified**: Exact match mirror confirmed on Hugging Face (`SunidhiSriram/twcs`, 492.58 MB raw CSV) with identical 7-column schema:
  `tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id`
- **Storage Strategy**:
  - Full raw dataset (`twcs.csv`) is ignored from git tracking via `.gitignore`.
  - Cleaned brand subset and sample subsets are kept lightweight and reproducible.

---

## 3. Directory Layout & Missing Dependencies

- Created git repo (`git init`).
- Added `.gitignore` to prevent leaking raw 500MB CSV or local secrets.
- Missing dependencies: None blocking (all ML, NLP, FAISS, and Streamlit packages are already installed and operational).

---

## 4. Recommended Next Action

Proceed to **Phase 1: Data Acquisition**:
1. Setup directory structure: `data/raw`, `data/processed`, `data/sample`, `data/golden`.
2. Provide automated, reproducible download script to acquire the verified dataset.
3. Generate `data/README.md` documenting schema, licensing, and access.
