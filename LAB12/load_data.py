import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://bi:bi@localhost:5432/ntpd")

df = pd.read_csv("data/transactions.csv")
df.to_sql("transactions", engine, if_exists="replace", index=False)

print("Załadowano wierszy:", len(df))
