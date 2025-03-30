
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np


app = FastAPI(title="Sales Forcaset API")
model = xgb.Booster() # to load the model from a json
model.load_model("./src/models/saved/xgboost/xgb_model_v1.json")

class PredictionInput(BaseModel): 
    store: int
    item: int
    year: int
    month: int
    day: int
    day_of_week: int
    week_of_year: int
    month_sin: float
    month_cos: float
    day_of_week_sin: float
    day_of_week_cos: float
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_rolling_7: float
    sales_rolling_14: float
    sales_rolling_30: float

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(input: PredictionInput):
    input_dict = input.dict()
    df = pd.DataFrame([input_dict])

    dmatrix = xgb.DMatrix(df)

    prediction = model.predict(dmatrix)
    result = float(prediction[0])

    return {"prediction": result}