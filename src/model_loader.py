from __future__ import annotations
import hashlib
from pathlib import Path
import joblib
from config import MODEL_VERSION, MODEL_DIR

def model_path() -> Path:
    return Path(MODEL_DIR) / f"{MODEL_VERSION}.joblib"

def sha256_file(path : Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_model():
    path = model_path()
    if not path.exists():
        raise FileExistsError(f"Model file not found: {path}")
    return joblib.load(path)
