import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import os

def train_model(df: pd.DataFrame, target_col: str = "Exited"):
    
    x = df.drop(columns = [target_col])
    y = df[target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    THRESHOLD = 0.3
    
    model = XGBClassifier(
        n_estimators = 300,
        learning_rate = 0.1,
        max_depth = 6,
        random_state = 42,
        n_jobs = -1,
        eval_metric = "logloss",
        scale_pos_weight = scale_pos_weight
    )
    
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    mlflow.set_tracking_uri(f"sqlite://{project_root}/mlruns/mlflow.db")
    mlflow.set_experiment("Bank Churn - XGBoost")
    
    with mlflow.start_run():
        
        model.fit(x_train, y_train)
        
        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= THRESHOLD).astype(int)
        
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, pos_label=1)
        prec = precision_score(y_test, preds, pos_label = 1)
        f1 = f1_score(y_test, preds, pos_label = 1)
        auc = roc_auc_score(y_test, proba)
        
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")        