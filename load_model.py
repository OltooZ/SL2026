import joblib
import numpy as np

# wczytanie zapisanego modelu
model = joblib.load("model_v1.joblib")

# przykładowe dane (4 cechy kwiatu iris)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# wykonanie predykcji
prediction = model.predict(sample)

print("Prediction:", prediction)