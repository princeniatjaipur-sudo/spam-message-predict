from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_DATASET_PATH, DEFAULT_DATASET_URL, DATA_DIR, SEED


def download_dataset(url: str = DEFAULT_DATASET_URL, destination_dir: Path = DATA_DIR) -> Path:
    """Download the UCI SMS Spam Collection dataset and extract it to the data folder."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / "smsspamcollection.zip"

    try:
        with request.urlopen(url, timeout=30) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to download the dataset from {url}: {exc}") from exc

    archive_path.write_bytes(archive_bytes)

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            archive.extractall(destination_dir)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Downloaded file is not a valid ZIP archive: {archive_path}") from exc

    extracted_path = destination_dir / "SMSSpamCollection"
    if not extracted_path.exists():
        for candidate in destination_dir.glob("*SMSSpamCollection*"):
            if candidate.is_file():
                extracted_path = candidate
                break

    if not extracted_path.exists():
        raise FileNotFoundError(f"Dataset was downloaded but the expected file was not found in {destination_dir}.")

    return extracted_path


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset schema and normalize basic data quality requirements."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty or missing.")

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if "label" not in cleaned.columns:
        if "Label" in cleaned.columns:
            cleaned = cleaned.rename(columns={"Label": "label"})
        else:
            raise ValueError("Dataset is missing the required 'label' column.")

    if "message" not in cleaned.columns:
        if "text" in cleaned.columns:
            cleaned = cleaned.rename(columns={"text": "message"})
        elif "Text" in cleaned.columns:
            cleaned = cleaned.rename(columns={"Text": "message"})
        else:
            raise ValueError("Dataset is missing the required 'message' column.")

    normalized = cleaned[["label", "message"]].copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    normalized["message"] = normalized["message"].fillna("").astype(str).str.strip()

    if normalized.empty:
        raise ValueError("Dataset has no rows after validation.")

    if normalized["message"].eq("").any():
        normalized = normalized[normalized["message"] != ""].copy()
    if normalized.empty:
        raise ValueError("Dataset contains no valid message content.")

    valid_labels = {"ham", "spam"}
    invalid_labels = sorted(normalized.loc[~normalized["label"].isin(valid_labels), "label"].unique())
    if invalid_labels:
        raise ValueError(
            "Dataset labels must be either 'ham' or 'spam'. Found: "
            + ", ".join(invalid_labels[:10])
        )

    normalized = normalized.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    return normalized


def load_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the dataset from disk or download it if not present."""
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        dataset_path = download_dataset(destination_dir=dataset_path.parent)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")

    try:
        if dataset_path.suffix == ".zip":
            with zipfile.ZipFile(dataset_path) as archive:
                members = archive.namelist()
                if not members:
                    raise ValueError("Downloaded dataset archive is empty.")
                target = next((m for m in members if m.endswith("SMSSpamCollection")), members[0])
                with archive.open(target) as handle:
                    df = pd.read_csv(handle, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
        else:
            df = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file {dataset_path} is empty or unreadable.") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read dataset file {dataset_path}: {exc}") from exc

    return validate_dataset(df)
