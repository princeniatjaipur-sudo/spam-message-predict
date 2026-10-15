from __future__ import annotations

from typing import Any


def clamp(value: float, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, int(round(value))))


def compute_risk_score(prediction: str, confidence: float, signals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Combine model confidence and transparent heuristic signals into a 0-100 risk score."""
    signal_score = 0
    if signals:
        for signal in signals:
            signal_score += int(signal.get("severity", 1)) * 8

    model_component = float(confidence) * 60.0
    heuristic_component = min(40.0, float(signal_score))
    combined = model_component + heuristic_component
    if prediction == "ham":
        combined = max(0.0, combined - 25.0)
    risk = clamp(combined)

    if risk >= 70:
        risk_level = "high"
    elif risk >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"

    explanation = (
        "The overall risk score combines the model's confidence with suspicious-language indicators. "
        "It is not a statistically calibrated probability of fraud; it is a transparent summary for review."
    )

    return {"score": risk, "risk_level": risk_level, "explanation": explanation}
