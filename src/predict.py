from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.explainability import explain_message
from src.risk_scoring import compute_risk_score
from src.train_model import load_model_bundle


def predict_single_message(message: str, model_path: str | Path = "models/spam_model.joblib") -> dict[str, Any]:
    """Predict a single message using the trained SpamShield model."""
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict([message])[0]
    probabilities = pipeline.predict_proba([message])[0]
    classes = pipeline.named_steps["classifier"].classes_
    if len(classes) == 2:
        prob_ham = float(probabilities[list(classes).index("ham")]) if "ham" in classes else 0.0
        prob_spam = float(probabilities[list(classes).index("spam")]) if "spam" in classes else 0.0
    else:
        prob_ham = float(probabilities[0])
        prob_spam = float(probabilities[1]) if len(probabilities) > 1 else 0.0

    if prediction == "spam":
        confidence = prob_spam
    else:
        confidence = prob_ham

    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out().tolist()
    explanation = explain_message(message, model=pipeline.named_steps["classifier"], feature_names=feature_names)
    risk_score = compute_risk_score(prediction, confidence, explanation["signals"])

    prediction_id = str(uuid.uuid4())
    result = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "message": message,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "probability_spam": round(float(prob_spam), 4),
        "probability_ham": round(float(prob_ham), 4),
        "risk_score": int(risk_score["score"]),
        "risk_level": risk_score["risk_level"],
        "signals": explanation["signals"],
        "top_terms": explanation["top_terms"],
        "summary": explanation["summary"],
    }
    return result


def batch_predict(messages: list[str], model_path: str | Path = "models/spam_model.joblib") -> list[dict[str, Any]]:
    """Predict a batch of messages and return list of result dictionaries."""
    return [predict_single_message(msg, model_path=model_path) for msg in messages if isinstance(msg, str) and msg.strip()]


def analyze_message(message: str, model_path: str | Path = "models/spam_model.joblib") -> dict[str, Any]:
    """Backward-compatible alias for single-message prediction."""
    return predict_single_message(message, model_path=model_path)
