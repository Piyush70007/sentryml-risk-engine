import pandas as pd

df = pd.read_csv("/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/data/raw/telco_churn.csv")

df["TotalCharges"]= pd.to_numeric(df["TotalCharges"],errors="coerce")

print("Dtype:",df["TotalCharges"].dtype)

print("Unique problematic rows:")
print(df[df["TotalCharges"]== " "].head())

df["TotalCharges"]= pd.to_numeric(df["TotalCharges"],errors="coerce")
