import pandas as pd
import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytanie danych
df = pd.read_csv("data/new_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Trenowanie modelu
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Predykcja
preds = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, preds)

print("Accuracy:", accuracy)

# Timestamp
timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

# Ścieżka modelu
model_path = (
    f"models/archive/"
    f"rf_model_{timestamp}.pkl"
)

# Zapis modelu
joblib.dump(model, model_path)

print("Model saved:", model_path)

# Aktualizacja modelu produkcyjnego
if accuracy >= 0.7:

    production_path = (
        "models/production/"
        "production_model.pkl"
    )

    joblib.dump(model, production_path)

    print("Production model updated")

else:
    print("Model rejected")