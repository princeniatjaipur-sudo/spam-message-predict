from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from config import HISTORY_DB_PATH


def get_connection(db_path: str | Path = HISTORY_DB_PATH) -> sqlite3.Connection:
    """Create or open a SQLite database connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db(db_path: str | Path = HISTORY_DB_PATH) -> None:
    """Create the prediction history table."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                message TEXT,
                label TEXT,
                confidence REAL,
                risk_score INTEGER,
                risk_level TEXT,
                indicators TEXT,
                metadata TEXT
            )
            """
        )
        conn.commit()


def save_prediction_record(record: dict[str, Any], db_path: str | Path = HISTORY_DB_PATH) -> None:
    """Store a prediction originating from a user or batch run."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO prediction_history (
                id, timestamp, message, label, confidence, risk_score, risk_level, indicators, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["timestamp"],
                record["message"],
                record["label"],
                float(record.get("confidence", 0.0)),
                int(record.get("risk_score", 0)),
                record.get("risk_level", "unknown"),
                json.dumps(record.get("indicators", [])),
                json.dumps(record.get("metadata", {})),
            ),
        )
        conn.commit()


def fetch_history(db_path: str | Path = HISTORY_DB_PATH, search: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Return saved prediction records, optionally filtered by text search."""
    query = "SELECT * FROM prediction_history"
    params: list[Any] = []
    if search:
        query += " WHERE message LIKE ? OR label LIKE ?"
        term = f"%{search}%"
        params.extend([term, term])
    query += " ORDER BY timestamp DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "message": row["message"],
                "label": row["label"],
                "confidence": row["confidence"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "indicators": json.loads(row["indicators"] or "[]"),
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]


def delete_history(db_path: str | Path = HISTORY_DB_PATH, ids: list[str] | None = None) -> None:
    """Delete one or more prediction records."""
    if not ids:
        with get_connection(db_path) as conn:
            conn.execute("DELETE FROM prediction_history")
            conn.commit()
        return

    placeholders = ", ".join("?" for _ in ids)
    with get_connection(db_path) as conn:
        conn.execute(f"DELETE FROM prediction_history WHERE id IN ({placeholders})", ids)
        conn.commit()


def count_history(db_path: str | Path = HISTORY_DB_PATH) -> int:
    """Count total records in the local history table."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) as total FROM prediction_history").fetchone()
        return int(row["total"])
