import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/Customer-Churn-Records.csv"
OUT = "data/processed/bank_churn_processed.csv"

df = pd.read_csv(RAW)

df = preprocess_data(df, target_col="Exited")

if "Exited" in df.columns and df["Exited"].dtype == "object":
    df["Exited"] = df["Exited"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

assert df["Exited"].isna().sum() == 0, "Exited has NaNs after preprocess"
assert set(df["Exited"].unique()) <= {0, 1}, "Exited not 0/1 after preprocess"

df_processed = build_features(df, target_col="Exited")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"Processed dataset saved to {OUT} | Shape: {df_processed.shape}")