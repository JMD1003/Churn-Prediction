import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(x,y):
    
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
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
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight
        }
        
        model = XGBClassifier(**params)
        scores = cross_val_score(model, x, y, cv = 5, scoring = "recall")
        return scores.mean()
    
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials=30)
    
    print("Best Params:", study.best_params)
    print("Best Recall:", study.best_value)
    return study.best_params