from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from config import DB_URL
import json
from pathlib import Path

ENGINE = create_engine(DB_URL, future=True)


def now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def upsert_customer(conn, row: dict):
  conn.execute(
    text("""
        INSERT OR REPLACE INTO customers (
          customerid, gender, seniorcitizen, partner, dependents, tenure, phoneservice,
          multiplelines, internetservice, onlinesecurity, onlinebackup, deviceprotection,
          techsupport, streamingtv, streamingmovies, contract, paperlessbilling,
          paymentmethod, monthlycharges, totalcharges, churn
        ) VALUES (
          :customerid, :gender, :seniorcitizen, :partner, :dependents, :tenure, :phoneservice,
          :multiplelines, :internetservice, :onlinesecurity, :onlinebackup, :deviceprotection,
          :techsupport, :streamingtv, :streamingmovies, :contract, :paperlessbilling,
          :paymentmethod, :monthlycharges, :totalcharges, NULL
        )
        """),
        row,
  )


def insert_prediction(conn, customerid: str, model_version: str, prob: float, pred: int):
  conn.execute(
    text("""
        INSERT INTO predictions (
          customerid, model_version, churn_probability, churn_prediction, created_at
        ) VALUES (
          :customerid, :model_version, :prob, :pred, :created_at
        )
        """),
        {
          "customerid": customerid,
          "model_version": model_version,
          "prob": prob,
          "pred": pred,
          "created_at": now_iso(),
        },
  )


def get_prediction_stats(conn, hours: int = 24) -> dict:

  q = text("""
        SELECT
          COUNT(*) AS n,
          AVG(churn_probability) AS avg_p,
          SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) AS churn_pred
        FROM predictions
        WHERE julianday(created_at) >= julianday('now') - (:hours / 24.0)
      """)

  row = conn.execute(q, {"hours": hours}).fetchone()

  n = int(row[0] or 0)
  avg_p = float(row[1] or 0.0)
  churn_pred = int(row[2] or 0)

  churn_rate = (churn_pred / n) if n > 0 else 0.0

  return {
    "window_hours": hours,
    "total_predictions": n,
    "avg_churn_probability": avg_p,
    "churn_predictions": churn_pred,
    "predicted_churn_rate": churn_rate,
  }

def get_latest_drift_status() -> dict:
    reports_dir = Path("monitoring/reports")
    if not reports_dir.exists():
        return {"drift_status": "unknown", "drift_report": None}

    files = sorted(reports_dir.glob("drift_*.json"))
    if not files:
        return {"drift_status": "unknown", "drift_report": None}

    latest = files[-1]
    report = json.loads(latest.read_text(encoding="utf-8"))

    return {
        "drift_status": report.get("overall_status", "unknown"),
        "drift_report": latest.name,
    }

def get_model_health(conn, hours: int = 24) -> dict:
    q = text("""
        SELECT
            COUNT(*) AS n,
            AVG(churn_probability) AS avg_p,
            SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) AS churn_pred
        FROM predictions
        WHERE julianday(created_at) >= julianday('now') - (:hours / 24.0)
    """)

    row = conn.execute(q, {"hours": hours}).fetchone()

    n = int(row[0] or 0)
    avg_p = float(row[1] or 0.0)
    churn_pred = int(row[2] or 0)
    churn_rate = (churn_pred / n) if n > 0 else 0.0

    status = "OK"
    alerts = []

    if n < 5:
        status = "WARN"
        alerts.append("Too few predictions in the selected window.")

    if n >= 20 and churn_rate > 0.60:
        status = "ALERT"
        alerts.append("Predicted churn rate unusually high (>60%).")

    if n >= 20 and churn_rate < 0.05:
        status = "ALERT"
        alerts.append("Predicted churn rate unusually low (<5%).")

    return {
        "window_hours": hours,
        "status": status,
        "total_predictions": n,
        "avg_churn_probability": avg_p,
        "churn_predictions": churn_pred,
        "predicted_churn_rate": churn_rate,
        "alerts": alerts,
    }