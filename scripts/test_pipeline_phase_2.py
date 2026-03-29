import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
from xgboost import XGBClassifier
import optuna
import sys, os
sys.path.append(os.path.abspath("src"))

from features.build_features import build_features

print("=== Phase 2: Modeling with XGBoost ===")

df = pd.read_csv("data/processed/bank_churn_processed.csv")

# Apply feature engineering to encode Geography and Card Type
df = build_features(df, target_col="Exited")

# Sanity checks
assert "Exited" in df.columns, "Target column missing!"
assert df["Exited"].isna().sum() == 0, "Exited has NaNs"
assert set(df["Exited"].unique()) <= {0, 1}, "Exited not 0/1"
assert df.select_dtypes(include="object").empty, "Object columns still present!"

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.3

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return recall_score(y_test, y_pred, pos_label=1)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)

# Final model with best params
print("\n=== Training final model with best params ===")
best_params = study.best_params
best_params.update({
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    "eval_metric": "logloss"
})

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)

print(classification_report(y_test, y_pred, digits=3))