import joblib
import pandas as pd

pipe = joblib.load("/Users/piyushchavan/Documents/crashcourse/ai-risk-intelligence/models/xgb_v4.joblib")

prep = pipe.named_steps["prep"]
features_names= prep.get_feature_names_out()

model = pipe.named_steps["model"]
importances = model.feature_importances_

fi = (
    pd.DataFrame({"features": features_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 20 features:")
print(fi.head(20).to_string(index=False))