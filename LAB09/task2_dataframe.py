import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg

# konfiguracja Pythona dla Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# obejście problemu Hadoop na Windows
os.environ["HADOOP_HOME"] = "C:/tmp"
os.environ["hadoop.home.dir"] = "C:/tmp"

spark = SparkSession.builder \
    .appName("DataFrameExample") \
    .master("local[*]") \
    .config("spark.python.worker.reuse", "false") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .getOrCreate()

# wczytanie CSV
df = spark.read.csv(
    "data/sales.csv",
    header=True,
    inferSchema=True
)

print("=== CALY DATAFRAME ===")
df.show()

print("=== SCHEMAT ===")
df.printSchema()

print("=== WYBRANE KOLUMNY ===")
df.select("produkt", "cena").show()

print("=== FILTROWANIE (cena > 500) ===")
df.filter(df.cena > 500).show()

print("=== GRUPOWANIE I AGREGACJE ===")
df.groupBy("kategoria").agg(
    sum("ilosc").alias("suma_ilosci"),
    avg("cena").alias("srednia_cena")
).show()

# zapis danych
try:
    df.coalesce(1).write.mode("overwrite").option("header", True).csv("sales_output_csv")
    df.write.mode("overwrite").parquet("sales_output_parquet")
    print("Pliki zapisane poprawnie!")
except Exception as e:
    print("Blad zapisu:", e)

spark.stop()