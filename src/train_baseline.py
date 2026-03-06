import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

DB_PATH = "/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/db/app.db"

engine =create_engine(f"sqlite:///{DB_PATH}")
df = pd.read_sql("SELECT * FROM customers", engine)

y = (df["churn"].astype(str).str.lower() == "yes").astype(int)

X = df.drop(columns=["churn","customerid"])

num_cols= X.select_dtypes(include=["int64", "float"]).columns.tolist()
cats_cols= [c for c in X.columns if c not in num_cols]

preprocess= ColumnTransformer(
    transformers=[
        ("num", "passthrough",num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"),cats_cols),
    ]
)

model = RandomForestClassifier(random_state= 42,n_estimators=300,class_weight="balanced")

pipe = Pipeline([
    ("prep",preprocess),
    ("model", model)
])

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("f1_score:",f1_score(y_test,pred))

joblib.dump(pipe, "models/baseline_rf.joblib")
print("saved model: models/baseline_rf.joblib")


