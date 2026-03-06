from datetime import datetime
from sqlalchemy import create_engine, text
from config import DB_URL , MODEL_VERSION

engine = create_engine(DB_URL)

def log_prediction(prob: float, pred: int) -> None:
    now = datetime.now().isoformat()
    q=text("""
           INSERT INTO predictions (model_version, churn_probability, churn_prediction, created_at)
           VALUES (:mv, :p, :y, :t)
    """)
    with engine.connect() as conn:
        conn.execute(q, {"mv": MODEL_VERSION, "p": prob, "y": pred, "t": now})
        conn.commit()

def get_prediction_stats(hours: int = 24) -> dict:
    window = f"-{hours} hours"
    q=text("""
           SELECT
               COUNT(*) as n,
               AVG(churn_probability) as avg_p,
               SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) as churn_pred
           FROM predictions
           WHERE created at >= datetime('now', :window)
    """)

    with engine.connect() as conn:
        row = conn.execute(q, {"window": window}).fetchone()
    
    n = int(row[0] or 0)
    avg_p= float(row[1] or 0.0)
    churn_pred = int(row[2] or 0)

    churn_rate = (churn_pred / n) if n > 0 else 0.0

    return{
        "window_hours": hours,
        "total_predictions": n,
        "avg_churn_probability": avg_p,
        "churn_prediction":churn_pred,
        "predicted_churn_rate": churn_rate,
    }

def db_ok() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).fetchone()
            return True
    except Exception:
        return False