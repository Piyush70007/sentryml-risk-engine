import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,accuracy_score
import joblib
from xgboost import XGBClassifier
import train_smote as p

MODEL_VERSION = "xgb_v4"

engine = create_engine(f"sqlite:///{p.DB_PATH}")
df= pd.read_sql("SELECT * FROM customers",engine)

y = (df["churn"].astype(str).str.lower()=="yes").astype(int)

df["tenure_safe"] =df["tenure"].replace(0,1)
df["totalcharges"]=pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
df["avg_charge_per_month"] =df["totalcharges"] / df["tenure_safe"]
df["is_high_monthly"]= (df["monthlycharges"] >= 80).astype(int) 

X= df.drop(columns=["churn", "customerid"])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num","passthrough",num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols),
    ]
)

neg = (y_train==0).sum()
pos = (y_train==1).sum()
scale_pos_weight = neg/pos

model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)

pipe = Pipeline([
    ("prep",preprocess),
    ("model",model)
])

pipe.fit(X_train,y_train)
pred= pipe.predict(X_test)

acc= accuracy_score(y_test,pred)
f1= f1_score(y_test,pred)

print("MODEL_VERSION:",MODEL_VERSION)
print("Accuracy:",acc)
print("F1:",f1)

joblib.dump(pipe, f"models/{MODEL_VERSION}.joblib")
print(f"saved model: models/{MODEL_VERSION}.joblib")