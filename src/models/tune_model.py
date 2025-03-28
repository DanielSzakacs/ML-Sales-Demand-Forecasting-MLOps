import optuna
import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def objective(trial, df, target="sales"):
    """
    Args: 
    Returns: 
    """
    df = df.dropna()
    features = [col for col in df.columns if col not in ["date", "sales", "prediction"]]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(y_test, y_pred))
    return rmse

def run_optuna_tuning(df, n_trials=30):
    """
    Args:
    Returns: 
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

    print("Best trials")
    print(study.best_trial)
    return study