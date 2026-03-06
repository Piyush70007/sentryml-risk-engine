import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import joblib


DB_PATH ="/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/db/app.db"
MODEL_VERSION = "rf_baseline_v1"
engine =create_engine(f"sqlite:///{DB_PATH}")

df = pd.read_sql("SELECT * FROM customers",engine)
X= df.drop(columns=["churn","customerid"])

pipe = joblib.load("/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/models/baseline_rf.joblib")

proba = pipe.predict_proba(X)[:,1]
pred = (proba >= 0.5).astype(int)

now = datetime.now().isoformat()

logs = pd.DataFrame({
    "customerid": df["customerid"],
    "model_version": MODEL_VERSION,
    "churn_probability": proba,
    "churn_prediction": pred,
    "created_at": now
})

logs.to_sql("predictions",con=engine, if_exists="replace",index=False)

print("logged predictions",len(logs))

with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT customerid , model_version, churn_probability, churn_prediction, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 5
    """)).fetchall()
    for r in rows:
        print(r)