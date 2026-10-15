from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET_PATH = DATA_DIR / "SMSSpamCollection"
DEFAULT_MODEL_PATH = MODELS_DIR / "spam_model.joblib"
HISTORY_DB_PATH = PROJECT_ROOT / "data" / "prediction_history.db"
FEEDBACK_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"
DEFAULT_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
MODEL_VERSION = "1.0.0"
SEED = 42
MAX_BATCH_ROWS = 200
MAX_FILE_SIZE_MB = 10
MAX_MESSAGE_LENGTH = 2000

OUTPUT_COLUMNS = [
    "message",
    "label",
    "confidence",
    "risk_score",
    "risk_level",
    "heuristic_indicators",
    "prediction_id",
    "timestamp",
]
