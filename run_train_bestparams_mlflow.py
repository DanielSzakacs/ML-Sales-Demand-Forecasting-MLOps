import pandas as pd
import json 

from sklearn.model_selection import train_test_split
from src.models.train_model_mlflow import train_xgboost_with_mlflow


# Load the best params of Xgboost
with open("./src/models/saved/xgboost/xgb_optuna_best_params.json") as f : 
    best_params = json.load(f)

# Load the clean sorce data 
df = pd.read_csv("./data/processed/clean_sales_data.csv", parse_dates=["date"])
df = df.sort_values(by=["date", "store", "item"])

# Split data
features = [col for col in df.columns if col not in ["sales", "date", "prediction"]]
target = "sales"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train with MLflow
train_xgboost_with_mlflow(X_train, X_test, y_train, y_test, best_params)