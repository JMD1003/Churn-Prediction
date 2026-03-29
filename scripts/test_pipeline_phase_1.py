import os
import sys
import pandas as pd

sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

DATA_PATH = r"C:\Users\jaked\Downloads\Churn Prediction\data\Customer-Churn-Records.csv"
TARGET_COL = "Exited"

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.dtypes)

    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n[4] Running sanity checks...")
    assert TARGET_COL in df_features.columns, "Target column missing!"
    assert df_features[TARGET_COL].isna().sum() == 0, "NaNs found in target!"
    assert set(df_features[TARGET_COL].unique()) <= {0, 1}, "Target not binary!"
    assert "Gender" not in df_features.select_dtypes(include="object").columns, "Gender still object dtype!"
    assert "Surname" not in df_features.columns, "Surname not dropped!"
    assert "RowNumber" not in df_features.columns, "RowNumber not dropped!"
    print("✅ All sanity checks passed!")

    print("\n✅ Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()