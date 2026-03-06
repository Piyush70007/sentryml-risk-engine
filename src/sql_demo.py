from sqlalchemy import create_engine, text 
engine = create_engine("sqlite:///db/app.db") 
with engine.connect() as conn: 
    result = conn.execute(text(""" 
    SELECT churn, COUNT(*) as total
    FROM customers_raw
    GROUP BY churn
    """)) 
    for row in result: 
        print(row)
