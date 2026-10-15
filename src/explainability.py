from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


URGENT_PATTERNS = [
    "urgent",
    "immediately",
    "act now",
    "limited time",
    "today only",
    "call now",
    "must respond",
    "exclusive",
]

SUSPICIOUS_URL_PATTERNS = ["http://", "https://", "www.", ".bit.ly", ".com", ".ru", ".xyz"]

SENSITIVE_REQUEST_PATTERNS = [
    "password",
    "pin",
    "otp",
    "verification code",
    "bank account",
    "credit card",
    "ssn",
    "routing number",
    "login",
    "verify your account",
]

PROMOTIONAL_PATTERNS = [
    "free",
    "winner",
    "congratulations",
    "claim",
    "cash",
    "prize",
    "gift",
    "bonus",
]


def detect_heuristic_signals(message: str) -> list[dict[str, str | int]]:
    """Return transparent heuristic indicators based on suspicious lexicons and patterns."""
    if not message:
        return []
    text = str(message).lower()
    findings: list[dict[str, str | int]] = []

    for keyword in URGENT_PATTERNS:
        if keyword in text:
            findings.append({"indicator": "urgent_language", "evidence": keyword, "severity": 2})
    for token in SUSPICIOUS_URL_PATTERNS:
        if token in text:
            findings.append({"indicator": "suspicious_link", "evidence": token, "severity": 3})
    for keyword in SENSITIVE_REQUEST_PATTERNS:
        if keyword in text:
            findings.append({"indicator": "sensitive_request", "evidence": keyword, "severity": 3})
    for keyword in PROMOTIONAL_PATTERNS:
        if keyword in text:
            findings.append({"indicator": "promotional_language", "evidence": keyword, "severity": 2})

    if re.search(r"(?<!\w)\d{4,}(?!\w)", text):
        findings.append({"indicator": "many_digits", "evidence": "digits", "severity": 1})

    unique_indicators: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for item in findings:
        key = str(item["indicator"])
        if key not in seen:
            seen.add(key)
            unique_indicators.append(item)
    return unique_indicators


def get_top_spam_terms(model: Any, feature_names: list[str], top_n: int = 12) -> list[tuple[str, float]]:
    """Return the most informative spam-associated words using Naive Bayes log-probabilities."""
    if not hasattr(model, "feature_log_prob_"):
        return []
    log_probs = model.feature_log_prob_
    spam_log_probs = log_probs[1]
    ham_log_probs = log_probs[0]
    relevance = spam_log_probs - ham_log_probs
    ranking = sorted(enumerate(relevance), key=lambda x: x[1], reverse=True)
    top_terms = []
    for idx, score in ranking[:top_n]:
        if idx < len(feature_names):
            top_terms.append((feature_names[idx], float(score)))
    return top_terms


def explain_message(message: str, model: Any | None = None, feature_names: list[str] | None = None, top_n: int = 8) -> dict[str, Any]:
    """Combine heuristic indicators and model-informed features into a single explanation payload."""
    signals = detect_heuristic_signals(message)
    top_terms: list[tuple[str, float]] = []
    if model is not None and feature_names is not None:
        top_terms = get_top_spam_terms(model, feature_names, top_n=top_n)

    explanation = "This assessment combines model behavior with rule-based security indicators."
    if signals:
        explanation += " Suspicious text patterns were detected, but they do not prove malicious intent on their own."
    else:
        explanation += " No obvious security indicators were detected in this message."

    return {"signals": signals, "top_terms": top_terms, "summary": explanation}
