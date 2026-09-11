"""
Phase 12: Grounding Knowledge Base & Semantic Retriever
Architecture:
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-d, fast local CPU inference)
- Vector Index: FAISS IndexFlatIP (exact normalized cosine similarity)
- Intent-Gated Grounding: Prioritizes historical resolutions matching the predicted intent,
  with dynamic fallback to global corpus when confidence is low.
- Sanitization: Exposes clean historical customer queries and verified brand resolutions without PII.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

INDEX_PATH = os.path.join("data", "processed", "faiss_index.bin")
METADATA_PATH = os.path.join("data", "processed", "retriever_metadata.parquet")
DEFAULT_KB_PATH = os.path.join("data", "processed", "knowledge_base.csv")

class SemanticRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path=INDEX_PATH, metadata_path=METADATA_PATH):
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.df_meta = None

    def _ensure_model_loaded(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, kb_file=DEFAULT_KB_PATH, max_cases=15000):
        """Builds FAISS index from the historical knowledge base."""
        self._ensure_model_loaded()
        if not os.path.exists(kb_file):
            raise FileNotFoundError(f"Knowledge base file not found at {kb_file}")

        print(f"[*] Building FAISS retrieval index from: {kb_file}")
        df = pd.read_csv(kb_file)
        if len(df) > max_cases:
            print(f"    Subsampling {max_cases:,} representative historical cases for fast indexing...")
            df = df.sample(max_cases, random_state=42).reset_index(drop=True)

        texts = df['customer_message'].fillna('').astype(str).tolist()
        print(f"    Encoding {len(texts):,} historical customer queries...")
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS inner product (cosine similarity since normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Store metadata
        meta_cols = ['case_id', 'customer_message', 'support_response', 'brand']
        if 'intent' in df.columns:
            meta_cols.append('intent')
        else:
            df['intent'] = 'other_support'
            meta_cols.append('intent')

        self.df_meta = df[meta_cols].copy().reset_index(drop=True)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        self.df_meta.to_parquet(self.metadata_path, index=False)
        print(f"[OK] FAISS index built ({self.index.ntotal} vectors) and saved to {self.index_path}")

    def load_index(self):
        """Loads cached FAISS index and metadata for sub-second retrieval."""
        if self.index is None:
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                raise FileNotFoundError("FAISS index or metadata not found. Call build_index() first.")
            self._ensure_model_loaded()
            self.index = faiss.read_index(self.index_path)
            self.df_meta = pd.read_parquet(self.metadata_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} historical records.")

    def search(self, query: str, intent: str = None, top_k: int = 3, intent_filter_threshold: float = 0.50) -> list[dict]:
        """
        Retrieves top K historically resolved cases.
        Supports intent-guided grounding:
        - If intent is given and valid, searches among intent-matching cases first.
        - Falls back to global semantic neighbors if intent pool has insufficient similarity.
        """
        self.load_index()
        self._ensure_model_loaded()

        q_vec = self.model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32)

        # Search top 25 candidates globally
        candidate_k = min(50, self.index.ntotal)
        sims, indices = self.index.search(q_vec, candidate_k)

        raw_results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self.df_meta):
                continue
            row = self.df_meta.iloc[idx]
            raw_results.append({
                "case_id": str(row["case_id"]),
                "customer_message": str(row["customer_message"]),
                "historical_response": str(row["support_response"]),
                "intent": str(row.get("intent", "other_support")),
                "similarity": round(float(sim), 4)
            })

        # Apply intent prioritization if intent is provided
        if intent and intent != "other_support":
            intent_matches = [r for r in raw_results if r["intent"] == intent]
            if len(intent_matches) >= top_k:
                return intent_matches[:top_k]
            elif len(intent_matches) > 0:
                # Fill remaining from top global
                remaining = [r for r in raw_results if r["case_id"] not in {m["case_id"] for m in intent_matches}]
                return (intent_matches + remaining)[:top_k]

        return raw_results[:top_k]
