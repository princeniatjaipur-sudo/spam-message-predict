"""SpamShield AI source package."""

from .data_loader import load_dataset, validate_dataset
from .preprocessing import clean_message, normalize_label, prepare_dataset
from .predict import analyze_message, batch_predict

__all__ = [
    "load_dataset",
    "validate_dataset",
    "clean_message",
    "normalize_label",
    "prepare_dataset",
    "analyze_message",
    "batch_predict",
]
