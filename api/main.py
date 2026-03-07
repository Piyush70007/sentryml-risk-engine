import json
from pathlib import Path
from src.model_loader import load_model, model_path, sha256_file
from src.prediction_store import db_ok
from src.db_utils import ENGINE, upsert_customer, insert_prediction, get_prediction_stats, get_latest_drift_status, get_model_health
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
import joblib
import pandas as pd

from config import MODEL_VERSION, DECISION_THRESHOLD

app = FastAPI(title = "SentryML Risk Engine API")

MODEL = None

pipe = joblib.load(f"models/{MODEL_VERSION}.joblib")

class customerInput(BaseModel):
    customerid: Optional[str] = None
    gender : str
    seniorcitizen: int
    partner :str
    dependents: str
    tenure : int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection : str
    techsupport : str
    streamingtv :str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges : float
    totalcharges : float

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_path: str
    exists: bool
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class PredictionStatsResponse(BaseModel):
    window_hours: int
    total_predictions: int
    avg_churn_probability: float
    churn_predictions: int
    predicted_churn_rate: float


class DriftMonitoringResponse(BaseModel):
    drift_status: str
    drift_report: Optional[str] = None


class PredictionMonitoringResponse(BaseModel):
    window_hours: int
    status: str
    total_predictions: int
    avg_churn_probability: float
    churn_predictions: int
    predicted_churn_rate: float
    alerts: list[str]


class ModelHealthResponse(BaseModel):
    model_version: str
    prediction_monitoring: PredictionMonitoringResponse
    drift_monitoring: DriftMonitoringResponse


class PredictResponse(BaseModel):
    customerid: str
    churn_probability: float
    churn_prediction: int
    model_version: str
    threshold: float

class RootResponse(BaseModel):
    status: str
    message: str
    docs: str
    endpoints: list[str]


class MetricsResponse(BaseModel):
    status: str
    latest_report: Optional[str] = None
    model_version: Optional[str] = None
    run_date: Optional[str] = None
    overall_status: Optional[str] = None
    path: Optional[str] = None

@app.get("/health", response_model=HealthResponse)
def health():
    return{
        "status":"ok",
        "db_connected":db_ok(),
        "model_version":MODEL_VERSION,
    }

@app.on_event("startup")
def _load_model_on_startup():
    global MODEL
    MODEL = load_model()

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    path = model_path()
    info = {
        "model_version": MODEL_VERSION,
        "model_path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        info["sha256"] = sha256_file(path)
        info["size_bytes"] = path.stat().st_size
    return info

@app.get("/", response_model=RootResponse)
def root():
    return {
        "status": "ok",
        "message": "Welcome to the SentryML Risk Engine API.",
        "docs": "/docs",
        "endpoints": ["/health","/predict", "/model-info", "/metrics","/model-health", "/prediction-stats"]
    }

@app.get("/prediction-stats", response_model=PredictionStatsResponse)
def prediction_stats(hours: int = 24):
    try:
        with ENGINE.connect() as conn:
            return get_prediction_stats(conn, hours=hours)
    except Exception as e:
        return {"error": str(e), "where": "prediction_stats"}

@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    reports_dir = Path("monitoring/reports")

    if not reports_dir.exists():
        return {"status": "no_reports_dir", "path": str(reports_dir)}

    reports = sorted(reports_dir.glob("drift_*.json"), reverse=True)
    if not reports:
        return {"status": "no_reports_found"}

    latest = reports[0]
    data = json.loads(latest.read_text(encoding="utf-8"))

    return {
        "status": "ok",
        "latest_report": str(latest),
        "model_version": data.get("model_version"),
        "run_date": data.get("run_date"),
        "overall_status": data.get("overall_status"),
    }

@app.get("/model-health", response_model=ModelHealthResponse)
def model_health(hours : int =24):
    try:
        with ENGINE.connect() as conn:
            stats = get_model_health(conn, hours=hours)
        drift = get_latest_drift_status()
        return{
            "model_version": MODEL_VERSION,
            "prediction_monitoring": stats,
            "drift_monitoring": drift,
        }
    except Exception as e:
        return {"error": str(e), "where": "model_health"}
    
@app.post("/predict", response_model=PredictResponse)
def predict(inp: customerInput):
    customerid = inp.customerid or str(uuid4())

    data = inp.model_dump()
    data["customerid"] = customerid
    X = pd.DataFrame([{k: v for k, v in data.items() if k != "customerid"}])

    X["tenure_safe"] = X["tenure"].replace(0, 1)
    X["totalcharges"] = pd.to_numeric(X["totalcharges"], errors="coerce").fillna(0)
    X["avg_charge_per_month"] = X["totalcharges"] / X["tenure_safe"]
    X["is_high_monthly"] = (X["monthlycharges"] >= 80).astype(int)

    prob = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(prob >= DECISION_THRESHOLD)

    customer_row = {k: data[k] for k in [
        "customerid","gender","seniorcitizen","partner","dependents","tenure","phoneservice",
        "multiplelines","internetservice","onlinesecurity","onlinebackup","deviceprotection",
        "techsupport","streamingtv","streamingmovies","contract","paperlessbilling",
        "paymentmethod","monthlycharges","totalcharges"
    ]}

    try:
        with ENGINE.begin() as conn:
            upsert_customer(conn, customer_row)
            insert_prediction(conn, customerid, MODEL_VERSION, prob, pred)
    except Exception as e:
        return {"error": str(e), "where": "db_logging"}
    
    return {
        "customerid": customerid,
        "churn_probability": prob,
        "churn_prediction": pred,
        "model_version"   : MODEL_VERSION,
        "threshold"       : DECISION_THRESHOLD
    }
