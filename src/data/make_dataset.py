import pandas as pd 
import os 
from src.features.feature_engineering import create_all_features

df = pd.read_csv("./data/train.csv", parse_dates=["date"])

df = create_all_features(df)
os.makedirs("./data/processed", exist_ok=True)

df.to_csv("./data/processed/clean_sales_data.csv", index=False)
print(df.head())