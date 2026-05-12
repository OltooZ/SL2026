import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    random_state=42
)

df = pd.DataFrame(X)
df["target"] = y

df.to_csv("data/new_data.csv", index=False)

print("Dataset generated")