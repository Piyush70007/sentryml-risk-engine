import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, MODEL_VERSION

BASELINE_PATH = Path(f"monitoring/baselines/baseline_{MODEL_VERSION}.json")
REPORTS_DIR = Path("monitoring/reports")

NUM_COLS = ["tenure", "monthlycharges", "totalcharges"]
CAT_COLS = ["contract", "internetservice", "paymentmethod"]

NUM_REL_CHANGE_ALERT = 0.20
CAT_PROB_SHIFT_ALERT = 0.15
UNKNOWN = "UNKNOWN"


def rel_change(old: float, new: float) -> float:
    if old == 0:
        return float("inf") if new != 0 else 0.0
    return abs(new - old) / abs(old)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_current_data(source: str, csv_path: str | None) -> pd.DataFrame:
    if source == "db":
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM customers"), conn)
        return normalize_columns(df)

    if source == "csv":
        if not csv_path:
            raise ValueError("For source='csv', you must provide --csv-path")
        df = pd.read_csv(csv_path)
        return normalize_columns(df)

    raise ValueError("source must be either 'db' or 'csv'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_PATH}. Run: python src/build_baseline_stats.py"
        )
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    df = load_current_data(source=args.source, csv_path=args.csv_path)

    report = {
        "model_version": MODEL_VERSION,
        "source": args.source,
        "csv_path": args.csv_path,
        "run_date": None,
        "numeric": {},
        "categorical": {},
        "overall_status": "OK",
    }

    print(f"\n=== DRIFT CHECK ({args.source.upper()} vs Baseline) ===")

    print("\n[Numeric drift]")
    for c in NUM_COLS:
        cur_mean = float(pd.to_numeric(df[c], errors="coerce").mean())
        base_mean = float(baseline["numeric"][c]["mean"])
        change = rel_change(base_mean, cur_mean)

        status = "ALERT" if change >= NUM_REL_CHANGE_ALERT else "OK"

        report["numeric"][c] = {
            "baseline_mean": base_mean,
            "current_mean": cur_mean,
            "rel_change": change,
            "threshold": NUM_REL_CHANGE_ALERT,
            "status": status,
        }

        if status == "ALERT":
            report["overall_status"] = "ALERT"

        print(
            f"{c:14s} baseline_mean={base_mean:8.3f} current_mean={cur_mean:8.3f} "
            f"rel_change={change:6.1%} => {status}"
        )

    print("\n[Categorical drift]")
    for c in CAT_COLS:
        cur = df[c].fillna(UNKNOWN).value_counts(normalize=True)
        base = baseline["categorical"][c]

        categories = set(base.keys()) | set(cur.index.tolist())

        report["categorical"][c] = {"threshold": CAT_PROB_SHIFT_ALERT, "alerts": []}
        alerts = []

        for cat in sorted(categories):
            base_p = float(base.get(cat, 0.0))
            cur_p = float(cur.get(cat, 0.0))
            diff = abs(cur_p - base_p)

            if diff >= CAT_PROB_SHIFT_ALERT:
                alerts.append((cat, base_p, cur_p, diff))
                report["categorical"][c]["alerts"].append(
                    {
                        "category": cat,
                        "baseline_p": base_p,
                        "current_p": cur_p,
                        "diff": diff,
                    }
                )

        if alerts:
            report["overall_status"] = "ALERT"

        print(f"\n{c}:")
        if not alerts:
            print(" OK (no big shifts)")
        else:
            print(" ALERT shifts:")
            for cat, base_p, cur_p, diff in alerts[:10]:
                print(
                    f"   - {cat:20s} baseline={base_p:6.1%} current={cur_p:6.1%} diff={diff:6.1%}"
                )

    run_date = datetime.now().strftime("%Y-%m-%d")
    report["run_date"] = run_date

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = REPORTS_DIR / f"drift_{args.source}_{MODEL_VERSION}_{run_date}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved drift report -> {out_path}")


if __name__ == "__main__":
    main()