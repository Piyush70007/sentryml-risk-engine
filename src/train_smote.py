import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

MODEL_VERSION = "rf_smote_v3"
DB_PATH = "/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/db/app.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
df = pd.read_sql("SELECT * FROM customers",engine)

y = (df["churn"].astype(str).str.lower()=="yes").astype(int)

df["tenure_safe"]= df["tenure"].replace(0,1)
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
df["avg_charge_per_month"]= df["totalcharges"]/df["tenure_safe"]
df["is_high_monthly"]= (df["monthlycharges"] >= 80).astype(int)

X=df.drop(columns=["churn", "customerid"])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3 ,random_state=42, stratify=y)

num_cols =X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols= [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough",num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"),cat_cols),
    ]
)

pipe = ImbPipeline(steps =[
    ("prep",preprocess),
    ("smote",SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=400,random_state=42))
])

pipe.fit(X_train,y_train)

pred = pipe.predict(X_test)

acc = accuracy_score(y_test,pred)
f1 = f1_score(y_test,pred)

print("MODEL_VERSION:", MODEL_VERSION)
print("Accuracy:",acc)
print("f1_score:",f1)

joblib.dump(pipe, f"models/{MODEL_VERSION}.joblib")
print("saved model:", f"models/{MODEL_VERSION}.joblib")
