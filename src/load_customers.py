import pandas as pd 
from sqlalchemy import create_engine

df = pd.read_csv("/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/data/raw/telco_churn.csv")
DB_PATH = "/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/db/app.db"
df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

engine = create_engine(f"sqlite:///{DB_PATH}")

df.to_sql("customers", con=engine, if_exists="replace", index = False)

print("customers tables loaded")
print("ROws:", len(df))