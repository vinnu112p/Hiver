"""
Phase 11: Final Intent Classifier
Hybrid Confidence-Gated Architecture:
1. Fast Primary: Calibrated N-Gram TF-IDF + Logistic Regression
2. Fallback: Structured LLM prompt when primary classifier confidence < threshold
3. Resilient Validation: Enforces strict JSON schema and taxonomy membership; never crashes
"""
import os
import json
import logging
import joblib
import numpy as np
from src.preprocess import clean_customer_text

logger = logging.getLogger(__name__)

TAXONOMY_INTENTS = [
    "billing_subscription",
    "cancellation_refund",
    "account_access_security",
    "audio_playback_issue",
    "app_crash_technical",
    "library_playlist_content",
    "other_support"
]

DEFAULT_MODEL_PATH = os.path.join("data", "processed", "tfidf_model.joblib")

class IntentClassifier:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.vectorizer = None
        self.classifier = None
        self.labels = TAXONOMY_INTENTS
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.vectorizer = bundle["vectorizer"]
                self.classifier = bundle["classifier"]
                self.labels = list(self.classifier.classes_)
                logger.info(f"Loaded classifier model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")

    def train(self, texts, labels):
        """Train TF-IDF + Logistic Regression."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=12000,
            sublinear_tf=True
        )
        X_vec = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X_vec, labels)
        self.labels = list(self.classifier.classes_)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, self.model_path)
        logger.info(f"Saved trained classifier to {self.model_path}")

    def predict(self, text: str) -> dict:
        """
        Classifies incoming customer text.
        Returns:
            {
                "intent": str,
                "confidence": float,
                "source": "tfidf_primary" | "llm_fallback" | "rule_heuristic",
                "probabilities": dict
            }
        """
        clean_text = clean_customer_text(text)
        if not clean_text:
            return {
                "intent": "other_support",
                "confidence": 1.0,
                "source": "empty_input",
                "probabilities": {it: (1.0 if it == "other_support" else 0.0) for it in TAXONOMY_INTENTS}
            }

        if self.classifier is None or self.vectorizer is None:
            # Fallback heuristic if model not yet fitted
            return self._heuristic_fallback(clean_text)

        # Primary Fast Inference
        vec = self.vectorizer.transform([clean_text])
        probs = self.classifier.predict_proba(vec)[0]
        max_idx = int(np.argmax(probs))
        pred_intent = self.labels[max_idx]
        confidence = float(probs[max_idx])
        prob_dict = {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))}

        # If confident, accept primary model
        if confidence >= self.confidence_threshold:
            return {
                "intent": pred_intent,
                "confidence": round(confidence, 4),
                "source": "tfidf_primary",
                "probabilities": prob_dict
            }

        # Otherwise: Low confidence -> check for LLM fallback if configured
        llm_res = self._try_llm_fallback(clean_text)
        if llm_res is not None:
            return llm_res

        # If LLM not configured or failed, return primary with its calibrated confidence
        return {
            "intent": pred_intent,
            "confidence": round(confidence, 4),
            "source": "tfidf_low_confidence",
            "probabilities": prob_dict
        }

    def _try_llm_fallback(self, text: str):
        """Optional structured LLM fallback for ambiguous / low-confidence cases."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"You are a customer support intent classifier for Spotify.\n"
                f"Classify the following customer tweet into exactly ONE of these intents:\n"
                f"{json.dumps(TAXONOMY_INTENTS)}\n\n"
                f"Customer Tweet: \"{text}\"\n\n"
                f"Respond with strict JSON ONLY:\n"
                f'{{"intent": "<chosen_intent>", "confidence": <float_between_0_and_1>}}'
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
                timeout=5.0
            )
            raw = response.choices[0].message.content.strip()
            # Clean possible markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            intent = data.get("intent")
            conf = float(data.get("confidence", 0.75))
            if intent in TAXONOMY_INTENTS:
                return {
                    "intent": intent,
                    "confidence": round(min(1.0, max(0.0, conf)), 4),
                    "source": "llm_fallback",
                    "probabilities": {it: (conf if it == intent else round((1.0 - conf) / 6.0, 4)) for it in TAXONOMY_INTENTS}
                }
        except Exception as e:
            logger.debug(f"LLM fallback skipped/failed: {e}")
            return None

    def _heuristic_fallback(self, text: str) -> dict:
        t = text.lower()
        if any(w in t for w in ["charge", "charged", "billing", "bill", "subscription", "premium"]):
            it = "billing_subscription"
        elif any(w in t for w in ["cancel", "refund", "stop charging"]):
            it = "cancellation_refund"
        elif any(w in t for w in ["login", "password", "hacked", "account", "access"]):
            it = "account_access_security"
        elif any(w in t for w in ["play", "sound", "pause", "bluetooth", "music", "song"]):
            it = "audio_playback_issue"
        elif any(w in t for w in ["crash", "update", "freeze", "error", "bug"]):
            it = "app_crash_technical"
        elif any(w in t for w in ["playlist", "library", "album", "sync", "lyrics"]):
            it = "library_playlist_content"
        else:
            it = "other_support"

        return {
            "intent": it,
            "confidence": 0.65,
            "source": "rule_heuristic",
            "probabilities": {k: (0.65 if k == it else 0.05) for k in TAXONOMY_INTENTS}
        }
