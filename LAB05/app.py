from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

app = FastAPI()

# LOAD MODEL
try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
    print(f"Model type: {type(model)}")
except Exception as e:
    raise RuntimeError(f"Nie można załadować modelu: {e}")

# INPUT
class InputData(BaseModel):
    value: float = Field(..., gt=0, lt=1000)

# ROOT
@app.get("/")
def root():
    return {"message": "API działa"}

# PREDICT
@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict([[data.value]])
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# INFO
@app.get("/info")
def info():
    return {
        "model": str(type(model)),
        "input_type": "float",
        "output_type": "float",
        "description": "Model loaded from model.pkl",
        "endpoints": ["/predict", "/info", "/health", "/version"]
    }

# HEALTH
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# VERSION
@app.get("/version")
def version():
    return {"version": "1.0.0"}

@app.get("/config")
def config():
    return {
        "app_name": os.getenv("APP_NAME", "default-app")
    }