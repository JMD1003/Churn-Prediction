import os
import json
import pandas as pd
import glob
import xgboost as xgb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "serving", "models")

model_paths = glob.glob(os.path.join(MODELS_DIR, "m-*"))
if not model_paths:
    raise Exception(f"No models found in {MODELS_DIR}")

MODEL_DIR = os.environ.get("MODEL_DIR", max(model_paths, key=os.path.getmtime))
print(f"Using model from: {MODEL_DIR}")

try:
    model_file = os.path.join(MODEL_DIR, "artifacts", "model.ubj")
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    print(f"Model loaded successfully from {model_file}")
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

try:
    feature_file = os.path.join(BASE_DIR, "..", "artifacts", "feature_columns.json")
    if not os.path.exists(feature_file):
        feature_file = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts", "feature_columns.json"))

    with open(feature_file) as f:
        FEATURE_COLS = json.load(f)
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

BINARY_MAP = {
    "Gender": {"Female": 0, "Male": 1},
}

NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary", "HasCrCard",
    "IsActiveMember", "Complain", "Satisfaction_Score", "Point_Earned"
]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Satisfaction_Score": "Satisfaction Score",
        "Point_Earned": "Point Earned",
        "Card_Type": "Card Type"
    }
    df = df.rename(columns=rename_map)

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping).astype("Int64")
                .fillna(0).astype(int)
            )

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        result = int(preds[0])
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"