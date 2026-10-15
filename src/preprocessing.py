from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SEED


def normalize_label(label: str) -> str:
    """Normalize labels to the canonical values ham/spam."""
    if label is None:
        return "ham"
    normalized = str(label).strip().lower()
    if normalized in {"spam", "spam message", "spam_msg", "1"}:
        return "spam"
    if normalized in {"ham", "ham message", "0", "non-spam"}:
        return "ham"
    return "ham" if normalized.startswith("ham") else "spam"


def clean_message(message: str) -> str:
    """Clean text while preserving useful URLs, numbers, punctuation, and words."""
    if message is None:
        return ""
    text = str(message)
    text = text.replace("\r", " ").replace("\n", " ")

    url_tokens: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        url_tokens.append(match.group(0))
        return f"__URL_TOKEN_{len(url_tokens) - 1}__"

    text = re.sub(r"https?://\S+|www\.\S+", protect_url, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([,.!?;:])\s*", r"\1 ", text)
    text = re.sub(r"(?<!\w)/(?!\w)", " ", text)
    for idx, url in enumerate(url_tokens):
        text = text.replace(f"__URL_TOKEN_{idx}__", url)
    return text.strip()


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataset for training by cleaning, normalizing labels, and dropping duplicates."""
    if df is None or df.empty:
        raise ValueError("Cannot prepare an empty dataset.")

    cleaned = df.copy()
    cleaned.columns = [str(col).strip().lower() for col in cleaned.columns]
    if "label" not in cleaned.columns or "message" not in cleaned.columns:
        if "text" in cleaned.columns:
            cleaned = cleaned.rename(columns={"text": "message"})
        else:
            raise ValueError("Dataset must contain 'label' and 'message' columns.")

    cleaned["label"] = cleaned["label"].map(normalize_label)
    cleaned["message"] = cleaned["message"].map(clean_message)
    cleaned = cleaned[(cleaned["message"].str.len() > 0)]
    cleaned = cleaned.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    return cleaned


def tokenize_text(text: str) -> list[str]:
    """Simple tokenizer compatible with the TF-IDF pipeline."""
    if text is None:
        return []
    return re.findall(r"\b\w+(?:\.\w+)?\b|https?://\S+|\b\d+\b", str(text).lower())


def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Return class counts for a dataset."""
    counts = df["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in ["ham", "spam"]}
