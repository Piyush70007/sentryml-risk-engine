import json
from pathlib import Path
import pandas as pd
from config import MODEL_VERSION

BASELINE_PATH = Path(f"monitoring/baselines/baseline_{MODEL_VERSION}.json")
DATA_PATH = Path("data/raw/telco_churn.csv")

NUM_COLS = ["tenure", "monthlycharges", "totalcharges"]
CAT_COLS = ["contract", "internetservice", "paymentmethod"]

def main():
    df =pd.read_csv(DATA_PATH)

    df.columns = [c.strip().lower() for c in df.columns]

    for c in NUM_COLS:
        df[c] =pd.to_numeric(df[c], errors= "coerce")

    baseline = {"numeric": {}, "categorical": {}}

    for c in NUM_COLS:
        baseline["numeric"][c] = {
            "mean": float(df[c].mean()),
            "std": float(df[c].std()),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }

    for c in CAT_COLS:
        vc=df[c].fillna("UNKNOW").value_counts(normalize=True)
        baseline["categorical"][c]= {k: float(v) for k, v in vc.items()}

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Saved baseline stats -> {BASELINE_PATH}")

if __name__ == "__main__":
    main()