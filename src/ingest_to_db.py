import pandas as pd
from sqlalchemy import create_engine

CSV_PATH ="data/raw/telco_churn.csv"
DB_PATH = "db/app.db"

df =pd.read_csv(CSV_PATH)

df.columns = [c.strip().lower().replace(" ", "_")for c in df.columns]

engine = create_engine(f"sqlite:///{DB_PATH}")

df.to_sql("customers_raw", con=engine, if_exists="replace", index=False)

print("loaded rows", len(df))
print("columns:", df.columns.tolist()[:10], "...")
print("saved to:",DB_PATH)