"""
Phase 15: Unified Support Pipeline
End-to-end execution flow:
Customer Message
    ↓
Cleaning & Normalization (src.preprocess)
    ↓
Intent Classification (src.classifier)
    ↓
Intent-Guided Semantic Retrieval (src.retriever)
    ↓
Grounded Response Generation (src.generator)
    ↓
Multi-Signal Escalation Decision (src.escalation)
    ↓
Structured Decision & Response JSON

Usable via CLI:
python -m src.pipeline --message "I was charged twice for premium"
python -m src.pipeline --input data/sample/sample_queries.json
"""
import os
import sys
import json
import argparse
from src.preprocess import clean_customer_text
from src.classifier import IntentClassifier
from src.retriever import SemanticRetriever
from src.generator import GroundedGenerator
from src.escalation import evaluate_escalation

class SupportAgentPipeline:
    def __init__(self, classifier=None, retriever=None, generator=None):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or SemanticRetriever()
        self.generator = generator or GroundedGenerator()

    def process_message(self, message: str) -> dict:
        """Processes a single customer message through the entire pipeline."""
        clean_msg = clean_customer_text(message)

        # 1. Intent Classification
        clf_result = self.classifier.predict(clean_msg)
        pred_intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 2. Historical Retrieval
        try:
            retrieved_raw = self.retriever.search(clean_msg, intent=pred_intent, top_k=3)
        except Exception as e:
            retrieved_raw = []

        top_similarity = retrieved_raw[0]["similarity"] if retrieved_raw else 0.0

        # 3. Grounded Reply Generation
        gen_result = self.generator.generate(clean_msg, pred_intent, retrieved_raw)
        reply = gen_result["reply"]

        # 4. Escalation Evaluation
        esc_result = evaluate_escalation(
            customer_message=message,
            predicted_intent=pred_intent,
            classifier_confidence=confidence,
            retrieval_similarity=top_similarity
        )

        formatted_cases = [
            {
                "case_id": c.get("case_id", ""),
                "similarity": c.get("similarity", 0.0),
                "customer": c.get("customer_message", ""),
                "agent": c.get("historical_response", "")
            }
            for c in retrieved_raw
        ]

        return {
            "message": message,
            "intent": {
                "label": pred_intent,
                "confidence": round(confidence, 4),
                "source": clf_result.get("source", "primary")
            },
            "retrieved_cases": formatted_cases,
            "reply": reply,
            "decision": esc_result["decision"],
            "reason": esc_result["reason"],
            "risk_level": esc_result.get("risk_level", "low"),
            "evidence_strength": round(top_similarity, 4),
            "grounding_summary": gen_result.get("grounding_summary", "")
        }

def main():
    parser = argparse.ArgumentParser(description="Run the Hiver AI Support Pipeline.")
    parser.add_argument("--message", type=str, help="Single customer message to analyze")
    parser.add_argument("--input", type=str, help="Path to JSON file containing sample queries")
    parser.add_argument("--output", type=str, help="Optional path to save output JSON")
    args = parser.parse_args()

    pipeline = SupportAgentPipeline()

    if args.message:
        res = pipeline.process_message(args.message)
        print(json.dumps(res, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data if isinstance(data, list) else data.get("queries", [])
        results = []
        for q in queries:
            text = q if isinstance(q, str) else q.get("text", "")
            results.append(pipeline.process_message(text))

        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    else:
        # Default interactive prompt
        print("Running default test query:")
        res = pipeline.process_message("I was charged twice for premium")
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
