import os
import sys

from pyspark.sql import SparkSession

# konfiguracja Spark + Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("RDDExample") \
    .master("local[*]") \
    .config("spark.python.worker.reuse", "false") \
    .getOrCreate()

sc = spark.sparkContext

# wczytanie pliku CSV jako RDD
rdd = sc.textFile("data/sales.csv")

print("=== CALY PLIK ===")
print(rdd.collect())

# pominięcie nagłówka
header = rdd.first()
data_rdd = rdd.filter(lambda row: row != header)

# split po przecinku
parsed_rdd = data_rdd.map(lambda line: line.split(","))

print("=== PRZETWORZONE DANE ===")
print(parsed_rdd.collect())

# filtrowanie produktów droższych niż 500
expensive_products = parsed_rdd.filter(
    lambda row: int(row[3]) > 500
)

print("=== PRODUKTY POWYZEJ 500 ===")
print(expensive_products.collect())

# liczba rekordów
count = parsed_rdd.count()
print(f"Liczba rekordow: {count}")

# suma cen
sum_prices = parsed_rdd.map(
    lambda row: int(row[3])
).reduce(lambda x, y: x + y)

print(f"Suma cen: {sum_prices}")

spark.stop()