import pandas as pd
import pytest

from src.data_loader import validate_dataset


def test_validate_dataset_accepts_valid_rows():
    df = pd.DataFrame({
        "label": ["ham", "spam"],
        "message": ["hello there", "WIN FREE CASH!"],
    })
    result = validate_dataset(df)
    assert list(result.columns) == ["label", "message"]
    assert len(result) == 2


def test_validate_dataset_rejects_invalid_labels():
    df = pd.DataFrame({
        "label": ["ham", "unknown"],
        "message": ["hello", "urgent offer"],
    })
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_validate_dataset_rejects_missing_columns():
    df = pd.DataFrame({"text": ["hello", "winner"]})
    with pytest.raises(ValueError):
        validate_dataset(df)
