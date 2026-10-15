from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, MODEL_VERSION, SEED
from src.data_loader import load_dataset
from src.evaluation import compute_metrics
from src.preprocessing import prepare_dataset


def build_pipeline(model_name: str = "naive_bayes") -> Pipeline:
    """Create a text classification pipeline with TF-IDF and a model."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20000,
    )
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    else:
        classifier = MultinomialNB(alpha=0.3)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def train_pipeline(df: pd.DataFrame, model_name: str = "naive_bayes") -> tuple[Pipeline, dict[str, Any]]:
    """Train a single model and return the pipeline and metrics."""
    prepared = prepare_dataset(df).copy()
    X = prepared["message"]
    y = prepared["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    pipeline = build_pipeline(model_name=model_name)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test.tolist(), y_pred.tolist())
    metrics["model_name"] = model_name
    metrics["dataset_shape"] = {"train": X_train.shape[0], "test": X_test.shape[0]}
    metrics["training_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return pipeline, metrics


def save_model_bundle(model: Pipeline, metadata: dict[str, Any], output_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Persist a trained model bundle with metadata for re-use."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": model, "metadata": metadata}
    joblib.dump(bundle, output_path)
    return {"path": str(output_path), "metadata": metadata}


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a persisted model bundle from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"The saved model at {model_path} is not a valid SpamShield model bundle.")
    return bundle


def train_default_model(dataset_path: str | Path = DEFAULT_DATASET_PATH, output_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Train the default SpamShield pipeline and save it to disk."""
    dataset = load_dataset(dataset_path)
    pipeline, metrics = train_pipeline(dataset, model_name="naive_bayes")
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_path": str(Path(dataset_path)),
        "training_timestamp": metrics["training_timestamp"],
        "model_type": "MultinomialNB",
        "metrics": metrics,
        "dataset_version": "UCI_SMS_Spam_Collection",
    }
    return save_model_bundle(pipeline, metadata, output_path=output_path)


def compare_classifiers(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, dict[str, Any]]:
    """Train the required NB model and a practical alternative classifier for comparison."""
    dataset = load_dataset(dataset_path)
    results = {}
    for name in ["naive_bayes", "logistic_regression"]:
        pipeline, metrics = train_pipeline(dataset, model_name=name)
        results[name] = metrics
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the SpamShield AI model.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Dataset path or file path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_PATH), help="Output path for the trained model")
    parser.add_argument("--force", action="store_true", help="Re-train even if a model already exists")
    parser.add_argument("--download", action="store_true", help="Download the dataset before training")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.download and not dataset_path.exists():
        from src.data_loader import download_dataset

        dataset_path = download_dataset(destination_dir=dataset_path.parent)

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Model already exists at {output_path}; use --force to retrain.")
        return

    result = train_default_model(dataset_path=dataset_path, output_path=output_path)
    print(json.dumps({"status": "trained", **result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
