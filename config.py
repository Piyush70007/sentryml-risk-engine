import os 

MODEL_VERSION = "xgb_v4"
DECISION_THRESHOLD = 0.4

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "db", "app.db")
DB_URL = f"sqlite:///{DB_PATH}"
MODEL_DIR = "models"