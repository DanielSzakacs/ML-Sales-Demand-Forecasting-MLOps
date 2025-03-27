import pandas as pd 
from src.models.train_model import train_base_model, train_xgboost_model

df = pd.read_csv("./data/processed/clean_sales_data.csv", parse_dates=["date"])
df = df.sort_values(by=["date", "store", "item"])

print("Train baseline model: ")
result = train_base_model(df)

print("Train xgboost model: ")
model , xgb_result = train_xgboost_model(df)
