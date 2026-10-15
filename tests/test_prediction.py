from __future__ import annotations

import pandas as pd
import pytest

from src.feedback import approve_feedback, list_feedback, record_feedback
from src.predict import batch_predict, predict_single_message
from src.train_model import train_pipeline


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "label": [
                "ham",
                "ham",
                "ham",
                "spam",
                "spam",
                "spam",
            ],
            "message": [
                "call me later",
                "meeting at 3pm",
                "can we meet tomorrow",
                "WIN free cash now",
                "urgent password reset",
                "claim your prize",
            ],
        }
    )


def test_training_pipeline_and_loading(sample_dataset, tmp_path):
    model, _ = train_pipeline(sample_dataset, model_name="naive_bayes")
    model_path = tmp_path / "sample_model.joblib"
    model_path.write_bytes(b"")
    assert model is not None
    assert hasattr(model, "predict")


def test_message_prediction_is_accurate_for_simple_cases(sample_dataset, tmp_path):
    model, _ = train_pipeline(sample_dataset, model_name="naive_bayes")
    path = tmp_path / "trained_model.joblib"
    import joblib

    joblib.dump({"pipeline": model, "metadata": {"metrics": {}}}, path)
    spam_result = predict_single_message("free prize winner", model_path=path)
    ham_result = predict_single_message("let's meet tomorrow", model_path=path)
    assert spam_result["prediction"] == "spam"
    assert ham_result["prediction"] == "ham"


def test_batch_predict_has_expected_length(sample_dataset, tmp_path):
    model, _ = train_pipeline(sample_dataset, model_name="naive_bayes")
    path = tmp_path / "batch_model.joblib"
    import joblib

    joblib.dump({"pipeline": model, "metadata": {"metrics": {}}}, path)
    results = batch_predict(["free prize now", "meet at lunch", "urgent verification code"], model_path=path)
    assert len(results) == 3
    assert {item["prediction"] for item in results} <= {"ham", "spam"}


def test_feedback_storage_and_history(tmp_path):
    feedback = record_feedback(
        message="claim free cash",
        prediction="ham",
        correction="spam",
        notes="incorrect predict",
        db_path=tmp_path / "feedback.db",
    )
    assert feedback["status"] == "pending"
    items = list_feedback(db_path=tmp_path / "feedback.db")
    assert len(items) == 1
    approve_feedback(feedback["id"], db_path=tmp_path / "feedback.db")
    items_after = list_feedback(db_path=tmp_path / "feedback.db", status="approved")
    assert len(items_after) == 1
