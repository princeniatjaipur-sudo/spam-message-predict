from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> dict[str, float | dict[str, int]]:
    """Compute key classification metrics for spam detection."""
    labels = ["ham", "spam"]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics


def summarize_evaluation(df: pd.DataFrame, y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> dict[str, Any]:
    """Return metrics plus false positives/false negatives details."""
    metrics = compute_metrics(y_true, y_pred)
    cm = np.array(metrics["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]
    results = {
        "metrics": metrics,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "records": df.to_dict(orient="records"),
    }
    return results


def build_confusion_df(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame for display."""
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    return pd.DataFrame(cm, index=["Actual Ham", "Actual Spam"], columns=["Predicted Ham", "Predicted Spam"])
