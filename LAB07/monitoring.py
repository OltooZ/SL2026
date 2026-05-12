import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import ClassificationPreset

# Dane historyczne
X_train, y_train = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    random_state=42
)

# Dane produkcyjne
X_prod, y_prod = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    random_state=999
)

columns = [f"feature_{i}" for i in range(5)]

# DataFrame train
df_train = pd.DataFrame(X_train, columns=columns)
df_train["target"] = y_train

# DataFrame production
df_prod = pd.DataFrame(X_prod, columns=columns)
df_prod["target"] = y_prod

# Model
model = RandomForestClassifier(random_state=42)

model.fit(
    df_train.drop("target", axis=1),
    df_train["target"]
)

# Predykcje
train_predictions = model.predict(
    df_train.drop("target", axis=1)
)

prod_predictions = model.predict(
    df_prod.drop("target", axis=1)
)

# Dodanie prediction
df_train["prediction"] = train_predictions
df_prod["prediction"] = prod_predictions

# Accuracy
train_acc = accuracy_score(
    df_train["target"],
    train_predictions
)

prod_acc = accuracy_score(
    df_prod["target"],
    prod_predictions
)

print("Train accuracy:", train_acc)
print("Production accuracy:", prod_acc)

# DATA DRIFT REPORT
drift_report = Report(
    metrics=[DataDriftPreset()]
)

drift_report.run(
    reference_data=df_train.drop(columns=["prediction"]),
    current_data=df_prod.drop(columns=["prediction"])
)

drift_report.save_html(
    "LAB07/data_drift_report.html"
)

print("Data drift report generated")

# CLASSIFICATION REPORT
classification_report = Report(
    metrics=[ClassificationPreset()]
)

classification_report.run(
    reference_data=df_train,
    current_data=df_prod
)

classification_report.save_html(
    "LAB07/classification_report.html"
)

print("Classification report generated")