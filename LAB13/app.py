import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# Konfiguracja wczytywana ze zmiennych środowiskowych (ConfigMap)
APP_NAME = os.getenv("APP_NAME", "ml-api")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

# MODEL
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)


class InputData(BaseModel):
    value: float


@app.get("/")
def root():
    return {"message": f"{APP_NAME} działa", "version": MODEL_VERSION}


@app.post("/predict")
def predict(data: InputData):
    if data.value is None:
        raise HTTPException(status_code=400, detail="Brak danych")

    prediction = model.predict([[data.value]])
    return {"prediction": float(prediction[0])}


@app.get("/info")
def info():
    return {
        "app": APP_NAME,
        "version": MODEL_VERSION,
        "model": "LinearRegression",
        "features": 1,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
