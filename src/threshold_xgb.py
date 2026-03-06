import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from config import DECISION_THRESHOLD

DB_PATH = "/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/db/app.db"

engine = create_engine(f"sqlite:///{DB_PATH}")
df = pd.read_sql("SELECT * FROM customers", engine)

y = (df["churn"].astype(str).str.lower() == "yes").astype(int)

df["tenure_safe"] = df["tenure"].replace(0, 1)
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
df["avg_charge_per_month"] = df["totalcharges"] / df["tenure_safe"]
df["is_high_monthly"] = (df["monthlycharges"] >= 80).astype(int)

X = df.drop(columns=["churn", "customerid"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = joblib.load("models/xgb_v4.joblib")

probs = pipe.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("Threshold | Precision | Recall | F1")
print("---------------------------------------")

for t in thresholds:
    preds = (probs >= t).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"{t:.2f}       | {precision:.3f}     | {recall:.3f} | {f1:.3f}")