from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
import numpy as np
from pydantic import BaseModel
from fastapi import HTTPException

app = FastAPI()

# MODEL
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

# INPUT
class InputData(BaseModel):
    value: float

# ROOT
@app.get("/")
def root():
    return {"message": "API działa"}

# PREDICT
@app.post("/predict")
def predict(data: InputData):
    if data.value is None:
        raise HTTPException(status_code=400, detail="Brak danych")

    prediction = model.predict([[data.value]])
    return {"prediction": float(prediction[0])}

# INFO
@app.get("/info")
def info():
    return {
        "model": "LinearRegression",
        "features": 1
    }

# HEALTH
@app.get("/health")
def health():
    return {"status": "ok"}