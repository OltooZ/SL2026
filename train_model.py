import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# wczytanie datasetu
iris = load_iris()

X = iris.data
y = iris.target

# utworzenie DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# analiza danych
print("Pierwsze wiersze danych:")
print(df.head())

print("\nRozmiar danych:")
print(df.shape)

print("\nInformacje o kolumnach:")
print(df.info())

# podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# stworzenie modelu
model = LogisticRegression(max_iter=1000)

# trenowanie
model.fit(X_train, y_train)

# predykcja
y_pred = model.predict(X_test)

# metryka
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy modelu:", accuracy)

# zapis modelu
joblib.dump(model, "model_v1.joblib")

print("\nModel zapisany jako model_v1.joblib")