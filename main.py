from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import json
app=FastAPI()
cat_models=joblib.load("cat_models.pkl")
xgb_models=joblib.load("xgb_models.pkl")


with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

@app.get("/")
def home():
    return {"message":"API is working"}

class InputData(BaseModel):
    data:dict


@app.post("/predict")
def predict(input_data:InputData):
    data=input_data.data
    df=pd.DataFrame([data])
    df=df.reindex(columns=feature_columns, fill_value=0)
    preds = {}
    mapping = {"adopted_within_07_days": "07","adopted_within_90_days": "90","adopted_within_120_days": "120"}

    for target,short in mapping.items():
        cat_pred = cat_models[target].predict_proba(df)[:, 1][0]
        xgb_pred = xgb_models[target].predict_proba(df)[:, 1][0]

        preds[short]=float(0.90 * cat_pred + 0.10 * xgb_pred)

    return {"predictions":preds}
    # return {"received_keys": list(data.keys())}