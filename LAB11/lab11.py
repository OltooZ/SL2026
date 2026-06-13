import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    round,
    sum,
    when,
    window,
    expr,
)

# ======================================================
# KONFIGURACJA SPARK (WINDOWS FIX)
# ======================================================

python_path = sys.executable

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
os.environ["PYTHONHASHSEED"] = "0"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = BASE_DIR / "checkpoint"

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

spark = (
    SparkSession.builder
    .appName("LAB11_STRUCTURED_STREAMING")
    .master("local[1]")
    .config("spark.python.worker.reuse", "false")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.network.timeout", "300s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ======================================================
# SCREEN 1
# ======================================================

print("=" * 60)
print("LAB11 - STRUCTURED STREAMING")
print("=" * 60)

print("Python:", sys.version.split()[0])
print("Spark:", spark.version)
print("Folder projektu:", BASE_DIR)

print("\n=== SCREEN 1 ===")
print("Wersja Python/Spark i poprawne uruchomienie środowiska")

# ======================================================
# ZADANIE 1 - STREAM SOURCE
# ======================================================

stream_df = (
    spark.readStream
    .format("rate")
    .option("rowsPerSecond", 5)
    .load()
)

# ======================================================
# ZADANIE 2 - TRANSFORMACJE
# ======================================================

transformed_df = (
    stream_df
    .withColumn(
        "category",
        when((col("value") % 4) == 0, "books")
        .when((col("value") % 4) == 1, "electronics")
        .when((col("value") % 4) == 2, "games")
        .otherwise("garden")
    )
    .withColumn(
        "amount",
        round((col("value") + 1) * 19.99, 2)
    )
    .withColumn(
        "status",
        when((col("value") % 5) == 0, "cancelled")
        .otherwise("paid")
    )
)

print("\nCzy DataFrame jest streamem?")
print(transformed_df.isStreaming)

print("\nSchemat danych:")
transformed_df.printSchema()

print("\n=== SCREEN 2 ===")
print("isStreaming=True + schema DataFrame")

# ======================================================
# ZADANIE 3 - AGREGACJE
# ======================================================

paid_df = transformed_df.filter(col("status") == "paid")

aggregated_df = (
    paid_df
    .groupBy("category")
    .agg(
        count("*").alias("events_count"),
        round(sum("amount"), 2).alias("total_amount")
    )
)

console_query = (
    aggregated_df.writeStream
    .format("console")
    .outputMode("complete")
    .option("truncate", False)
    .trigger(processingTime="5 seconds")
    .start()
)

print("\n=== SCREEN 3 ===")
print("Streaming agregacji w konsoli")

time.sleep(15)

console_query.stop()

# ======================================================
# ZADANIE 4 - WATERMARK + WINDOWS
# ======================================================

windowed_df = (
    paid_df
    .withWatermark("timestamp", "10 seconds")
    .groupBy(
        window(col("timestamp"), "10 seconds"),
        col("category")
    )
    .agg(
        count("*").alias("events_count"),
        round(sum("amount"), 2).alias("total_amount")
    )
)

window_query = (
    windowed_df.writeStream
    .format("console")
    .outputMode("update")
    .option("truncate", False)
    .trigger(processingTime="5 seconds")
    .option(
        "checkpointLocation",
        str(CHECKPOINT_DIR)
    )
    .start()
)

print("\n=== SCREEN 4 ===")
print("Watermark + window aggregation")

time.sleep(15)

window_query.stop()

# ======================================================
# ZADANIE 5 - ZAPIS WYNIKÓW
# ======================================================

file_query = (
    windowed_df.writeStream
    .format("parquet")
    .option(
        "path",
        str(OUTPUT_DIR)
    )
    .option(
        "checkpointLocation",
        str(CHECKPOINT_DIR / "parquet")
    )
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

print("\n=== SCREEN 5 ===")
print("Zapis streamu do plików")

time.sleep(10)

file_query.stop()

print("\nZapisane pliki:")

files = list(OUTPUT_DIR.glob("*"))

for file in files:
    print(file.name)

print("\n=== SCREEN 6 ===")
print("Pliki wyjściowe po streamingu")

spark.stop()

print("\nProgram zakończony poprawnie.")

print("\n=== SCREEN 7 ===")
print("Exit code 0 i poprawne zakończenie")