from sklearn.metrics import mean_squared_error

import numpy as np 
import mlflow
import mlflow.xgboost
import xgboost as xgb

def train_xgboost_with_mlflow(X_train, X_test, y_train, y_test, params):    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("Xgboost model v1")

    with mlflow.start_run():
    
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log paraméterek
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrika
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.xgboost.log_model(model, "xgboost_model")

        print(f"RMSE: {rmse:.2f}")
