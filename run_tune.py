import pandas as pd 
from src.models.tune_model import run_optuna_tuning
import json
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

df = pd.read_csv("./data/processed/clean_sales_data.csv", parse_dates=["date"])
df = df.sort_values(by=["date", "store", "item"])

study = run_optuna_tuning(df)
best_params = study.best_params

# Save best params
with open("./src/models/saved/xgb_optuna_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

# Plot and save the optuna optimalization and param importance
fig1 = plot_optimization_history(study)
fig1.write_html("reports/optuna_optimalization_history.html")
fig1.show()

fig2 = plot_param_importances(study)
fig2.write_html("reports/optuna_param_importance.html")
fig2.show()