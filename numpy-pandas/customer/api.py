from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans.pkl")

app = FastAPI()


class CustomerFeatures(BaseModel):
    total_spent: float
    num_orders: int
    avg_order: float
    days_since_last_order: int


@app.get("/predict")
def predict_segment(data: CustomerFeatures):
    X = np.array([[data.total_spent, data.num_orders, data.avg_order, data.days_since_last_order]])
    X_scaled = scaler.transform(X)
    cluster = int(model.predict(X_scaled)[0])
    return {"cluster": cluster}
