from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from config import MODEL_VERSION, DECISION_THRESHOLD

app = FastAPI(title = "AI Risk Intelligence - Churn API")

pipe = joblib.load(f"models/{MODEL_VERSION}.joblib")

class customerInput(BaseModel):
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

@app.get("/")
def health():
    return {"status": "running", "model_version": MODEL_VERSION, "threshold": DECISION_THRESHOLD}

@app.post("/predict")
def predict(inp: customerInput):
    X=pd.DataFrame([inp.dict()])

    X["tenure_safe"]= X["tenure"].replace(0,1)
    X["totalcharges"]= pd.to_numeric(X["totalcharges"],errors="coerce").fillna(0)
    X["avg_charge_per_month"]= X["totalcharges"] / X["tenure_safe"]
    X["is_high_monthly"]= (X["monthlycharges"] >= 80).astype(int)

    prob =float(pipe.predict_proba(X)[:, 1][0])
    pred = int(prob >= DECISION_THRESHOLD)

    return {
        "churn_probability": prob,
        "churn_prediction": pred,
        "model_version"   : MODEL_VERSION,
        "threshold"       : DECISION_THRESHOLD
    }



