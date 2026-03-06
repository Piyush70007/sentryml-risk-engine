from sqlalchemy import create_engine, text
import pandas as pd
from config import DB_URL

engine = create_engine(DB_URL)

def q(sql:str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)
    
def main():
    pred_stats = q("""
    SELECT
        COUNT(*) as n_predictions,
        AVG(churn_probability) as avg_churn_probability,
        AVG(churn_prediction) as predicted_churn_rate
    FROM predictions
    WHERE created_at >= datetime('now', '-1 day');
    """)
    print("\n=== Predictions (last 24h) ===")
    print(pred_stats.to_string(index=False))

    n = pred_stats.loc[0, "n_predictions"]
    if n == 0:
        print("\nALERT: No predictions logged in the last 24 hours.")
    
    contract_dist = q("""
    SELECT contract, COUNT(*) as n
    FROM customers 
    GROUP BY contract
    ORDER BY n DESC;
    """)

    print("\n=== Customers: contract distribution ===")
    print(contract_dist.to_string(index = False))

    num_stats = q("""
    SELECT
        AVG(monthlycharges) as avg_monthlycharges,
        MIN(monthlycharges) as min_monthlycharges,
        MAX(monthlycharges) as max_monthlycharges,
        AVG(tenure) as avg_tenure
    FROM customers;
    """)

    print("\n=== Customers: numeric stats ===")
    print(num_stats.to_string(index=False))

if __name__ =="__main__":
    main()