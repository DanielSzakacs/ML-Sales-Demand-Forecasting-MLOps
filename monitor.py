import xgboost as xgb
import pandas as pd 
import numpy as np

from sklearn.metrics import mean_squared_error
from src.features.feature_engineering import create_all_features
from datetime import datetime


model = xgb.Booster()
model.load_model("./src/models/saved/xgboost/xgb_model_v1.json")

# read the new test data
test = pd.read_csv("./data/test.csv", parse_dates=["date"])
sample_submission = pd.read_csv("./data/sample_submission.csv")
df = pd.merge(test, sample_submission, how="inner", on="id")
df.drop("id", axis=1, inplace=True)

# Feature 'pipline'
df = create_all_features(df)

# Split data 
features = [col for col in df.columns if col not in ["sales", "date", "prediction"]]
target = "sales"
X = df[features]
y = df[target]

# Predict
dmatrix = xgb.DMatrix(X)
y_pred = model.predict(dmatrix)

# RMSE count
rmse = np.sqrt(mean_squared_error(y, y_pred))

THRESHOLD = 20

print(f"[{datetime.now()}] RMSE on new data: {rmse:.2f}")

if rmse > THRESHOLD:
    print("Drift detected! RMSE above threshold.")

