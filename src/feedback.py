from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_MODEL_PATH, FEEDBACK_DB_PATH, SEED
from src.data_loader import load_dataset
from src.train_model import save_model_bundle, train_pipeline


def get_feedback_connection(db_path: str | Path = FEEDBACK_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db(db_path: str | Path = FEEDBACK_DB_PATH) -> None:
    with get_feedback_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                message TEXT,
                prediction TEXT,
                correction TEXT,
                notes TEXT,
                status TEXT
            )
            """
        )
        conn.commit()


def record_feedback(message: str, prediction: str, correction: str, notes: str = "", db_path: str | Path = FEEDBACK_DB_PATH) -> dict[str, Any]:
    init_feedback_db(db_path)
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_feedback_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback (id, timestamp, message, prediction, correction, notes, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (feedback_id, timestamp, message, prediction, correction, notes, "pending"),
        )
        conn.commit()
    return {"id": feedback_id, "timestamp": timestamp, "message": message, "prediction": prediction, "correction": correction, "notes": notes, "status": "pending"}


def list_feedback(db_path: str | Path = FEEDBACK_DB_PATH, status: str | None = None) -> list[dict[str, Any]]:
    init_feedback_db(db_path)
    query = "SELECT * FROM feedback"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY timestamp DESC"
    with get_feedback_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def approve_feedback(feedback_id: str, db_path: str | Path = FEEDBACK_DB_PATH) -> None:
    with get_feedback_connection(db_path) as conn:
        conn.execute("UPDATE feedback SET status = ? WHERE id = ?", ("approved", feedback_id))
        conn.commit()


def reject_feedback(feedback_id: str, db_path: str | Path = FEEDBACK_DB_PATH) -> None:
    with get_feedback_connection(db_path) as conn:
        conn.execute("UPDATE feedback SET status = ? WHERE id = ?", ("rejected", feedback_id))
        conn.commit()


def retrain_with_approved_feedback(
    dataset_path: str | Path,
    base_model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    db_path: str | Path = FEEDBACK_DB_PATH,
) -> dict[str, Any]:
    """Retrain using approved corrections, preserving the original dataset and the old model if performance drops."""
    init_feedback_db(db_path)
    original_df = load_dataset(dataset_path)
    approved = list_feedback(db_path, status="approved")
    if not approved:
        raise ValueError("No approved feedback examples are available for retraining.")

    additions = pd.DataFrame(
        [{"label": row["correction"].lower().strip(), "message": row["message"].strip()} for row in approved]
    )
    additions = additions[(additions["label"].isin(["ham", "spam"])) & (additions["message"].str.len() > 0)]
    combined = pd.concat([original_df, additions], ignore_index=True)
    combined = combined.drop_duplicates(subset=["message"], keep="last").reset_index(drop=True)

    baseline_pipeline, baseline_metrics = train_pipeline(original_df, model_name="naive_bayes")
    retrained_pipeline, retrained_metrics = train_pipeline(combined, model_name="naive_bayes")
    selected = retrained_metrics if retrained_metrics["f1"] >= baseline_metrics["f1"] else baseline_metrics
    model = baseline_pipeline if selected == baseline_metrics else retrained_pipeline
    model_version = "1.1.0"
    metadata = {
        "model_version": model_version,
        "training_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_version": "UCI_SMS_Spam_Collection",
        "feedback_approved": len(additions),
        "baseline_metrics": baseline_metrics,
        "retrained_metrics": retrained_metrics,
        "selected_model": "existing" if selected == baseline_metrics else "retrained",
    }
    save_model_bundle(model, metadata, output_path=output_path)
    return metadata
