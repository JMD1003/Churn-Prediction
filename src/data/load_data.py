import pandas as pd
import os

def load_data(filepath: str = r"C:\Users\jaked\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)