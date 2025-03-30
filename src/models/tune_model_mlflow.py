import optuna
import pandas as pd 
import numpy as np 
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def objective(trial, df, target="sales"):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): A single trial object containing the current set of hyperparameters.
        df (pd.DataFrame): The dataset containing features and the target variable.
        target (str, optional): The name of the target column. Defaults to "sales".

    Returns:
        float: Root Mean Squared Error (RMSE) of the model on the test set.
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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    with mlflow.start_run(nested=True):
        mlflow.log_params(param)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, "optuna_xgboost_model")

    return rmse

def run_optuna_tuning(df, n_trials=20):
    """
    Runs Optuna hyperparameter tuning for an XGBoost model.

    Args:
        df (pd.DataFrame): The dataset used for training and evaluation.
        n_trials (int, optional): Number of Optuna trials to run. Defaults to 3.

    Returns:
        optuna.study.Study: The study object containing optimization results.
    """
    mlflow.set_experiment("optuna_xgboost_tuning_20_trial")
    study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

    print("Best trials")
    print(study.best_trial)
    return study