import pandas as pd

from src.preprocessing import clean_message, normalize_label, prepare_dataset


def test_normalize_label_handles_variants():
    assert normalize_label("spam") == "spam"
    assert normalize_label("SPAM ") == "spam"
    assert normalize_label("ham") == "ham"
    assert normalize_label("non-spam") == "ham"


def test_clean_message_preserves_core_text():
    raw = "WIN!!! FREE  cash  now http://example.com"
    cleaned = clean_message(raw)
    assert "http://example.com" in cleaned
    assert "FREE" in cleaned
    assert "cash" in cleaned


def test_prepare_dataset_normalizes_and_deduplicates():
    df = pd.DataFrame({
        "label": ["ham", "Spam", "ham", "spam"],
        "message": ["hello", "FREE prize now", "hello", "urgent password reset"],
    })
    prepared = prepare_dataset(df)
    assert set(prepared["label"]) <= {"ham", "spam"}
    assert len(prepared) == 3
