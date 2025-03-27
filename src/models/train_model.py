from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os 

import numpy as np
import xgboost as xgb

def evaluate_model(y_true, y_pred):
    """
    Evalueate and print the model prediction based MSE MAE and R2

    Args: 
        y_true (Real data which the model should predict)
        y_pred (Predicted data)

    Returns: 
        Results of the MSE, MAE and R2 score tests
    """
    rmse_result = np.sqrt(mse(y_true=y_true, y_pred=y_pred))
    mae_result = mae(y_true=y_true, y_pred=y_pred)
    r2_result = r2_score(y_true=y_true, y_pred=y_pred)

    print(f"MSE score: {rmse_result:.2f}")
    print(f"MAE score: {mae_result:.2f}")
    print(f"R2 score: {r2_result:.2f}")

    return {"rmse": rmse_result, 
            "mae" : mae_result,
            "r2": r2_result}



def train_base_model(df):
    """
    Args:
    Returns: 
    """
    df = df.sort_values(by=["date", "store", "item"])
    df["prediction"] = df.groupby(["store", "item"])["sales"].shift(1)
    df = df.dropna()

    return evaluate_model(df["sales"], df["prediction"])


def train_xgboost_model(df, target="sales"):
    """
    Args:
    Returns:
    """

    # make sure there is no na
    df = df.dropna()
    # get teh features which we will use
    features = [col for col in df.columns if col not in ["sales", "date", "prediction"]]
    
    X = df[features]
    y = df[target]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Model 
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)

    os.makedirs("./src/models/saved/xgboost", exist_ok=True)
    model.save_model("./src/models/saved/xgboost/xgb_model_v1.json")

    return model, metrics