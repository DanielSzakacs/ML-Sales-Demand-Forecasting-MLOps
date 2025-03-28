import pandas as pd 
from src.models.tune_model import run_optuna_tuning

df = pd.read_csv("./data/processed/clean_sales_data.csv", parse_dates=["date"])
df = df.sort_values(by=["date", "store", "item"])

study = run_optuna_tuning(df)
print(study)