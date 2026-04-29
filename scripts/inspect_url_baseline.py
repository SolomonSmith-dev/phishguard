"""Quick diagnostic: print top feature importances."""
import json
from pathlib import Path
import lightgbm as lgb
import pandas as pd

booster = lgb.Booster(model_file="models/checkpoints/url_model.lgb")
feature_names = json.loads(Path("models/checkpoints/url_features.json").read_text())

importance = booster.feature_importance(importance_type="gain")
df = pd.DataFrame({"feature": feature_names, "gain": importance}).sort_values("gain", ascending=False)
total = df["gain"].sum()
df["pct"] = (df["gain"] / total * 100).round(2)

print("Top 15 features by gain:")
print(df.head(15).to_string(index=False))
print(f"\nTop 3 features account for {df.head(3)['pct'].sum():.1f}% of total gain")
