import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PARQUET_FILE = OUTPUT_DIR / "products.parquet"
QUERY_RESULT_FILE = OUTPUT_DIR / "query_result.csv"


def clean_output(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def print_screen_marker(number: int, description: str) -> None:
    print()
    print(f"=== SCREEN {number}: {description} ===")
    print()


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (
    SparkSession.builder
    .appName("SparkSQL_Parquet_CSV_LAB10")
    .master("local[1]")
    .config("spark.python.worker.reuse", "false")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext._jsc.hadoopConfiguration().set("io.native.lib.available", "false")

try:
    print("LAB10 - Analiza danych w Spark SQL")
    print("Projekt:", BASE_DIR)

    # Zadanie 1: przygotowanie danych Parquet, odczyt, show() i printSchema().
    print("\n" + "=" * 70)
    print("Zadanie 1: Przygotowanie srodowiska i podstawowe wczytanie pliku Parquet")
    print("=" * 70)

    products_data = [
        (101, "Laptop", "Electronics", 3200),
        (102, "Phone", "Electronics", 2400),
        (103, "Desk", "Furniture", 900),
        (104, "Chair", "Furniture", 450),
        (105, "Monitor", "Electronics", 1200),
        (106, "Printer", "Office", 1500),
    ]
    products_columns = ["product_id", "product_name", "category", "unit_price"]
    products_df = spark.createDataFrame(products_data, products_columns)

    OUTPUT_DIR.mkdir(exist_ok=True)
    clean_output(PARQUET_FILE)
    products_pdf = pd.DataFrame(products_data, columns=products_columns)
    pq.write_table(pa.Table.from_pandas(products_pdf), PARQUET_FILE)

    df_parquet = spark.read.parquet(str(PARQUET_FILE))

    print("Wczytane dane z pliku Parquet:")
    df_parquet.show()
    print("Struktura danych Parquet:")
    df_parquet.printSchema()
    print("Plik Parquet:", PARQUET_FILE)
    print_screen_marker(1, "Po Zadaniu 1: widoczny df_parquet.show(), printSchema() i plik products.parquet")

    # Zadanie 2: CSV, Temporary View, proste zapytanie SQL.
    print("\n" + "=" * 70)
    print("Zadanie 2: Ladowanie danych CSV i rejestracja jako widok tabelaryczny")
    print("=" * 70)

    sales_csv_path = DATA_DIR / "sales.csv"
    sales_df = (
        spark.read.csv(
            str(sales_csv_path),
            header=True,
            inferSchema=True,
        )
    )

    sales_df.createOrReplaceTempView("sales_view")

    print("Dane CSV - pierwsze wiersze:")
    sales_df.show(10)
    print("Struktura danych CSV:")
    sales_df.printSchema()

    print("Zapytanie SQL: SELECT * FROM sales_view LIMIT 10")
    spark.sql("SELECT * FROM sales_view LIMIT 10").show()
    print_screen_marker(2, "Po Zadaniu 2: widoczny CSV, Temporary View sales_view i wynik SELECT LIMIT 10")

    # Zadanie 3: Spark SQL - agregacje, GROUP BY, WHERE, JOIN, zapis wyniku.
    print("\n" + "=" * 70)
    print("Zadanie 3: Spark SQL. Tworzenie zapytan")
    print("=" * 70)

    df_parquet.createOrReplaceTempView("products_view")

    print("Agregacje: SUM, AVG, COUNT")
    aggregation_result = spark.sql(
        """
        SELECT
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_amount,
            ROUND(AVG(amount), 2) AS average_amount
        FROM sales_view
        """
    )
    aggregation_result.show()
    print_screen_marker(3, "Po agregacjach: COUNT, SUM i AVG")

    print("Grupowanie po region oraz product_id")
    grouped_result = spark.sql(
        """
        SELECT
            region,
            product_id,
            SUM(quantity) AS total_quantity,
            SUM(amount) AS total_amount
        FROM sales_view
        GROUP BY region, product_id
        ORDER BY region, product_id
        """
    )
    grouped_result.show()
    print_screen_marker(4, "Po GROUP BY region, product_id")

    print("Filtrowanie warunkowe: WHERE amount > 3000")
    filtered_result = spark.sql(
        """
        SELECT *
        FROM sales_view
        WHERE amount > 3000
        ORDER BY amount DESC
        """
    )
    filtered_result.show()
    print_screen_marker(5, "Po WHERE amount > 3000")

    print("JOIN dwoch widokow: sales_view + products_view")
    join_result = spark.sql(
        """
        SELECT
            s.sale_id,
            s.region,
            p.product_name,
            p.category,
            s.quantity,
            s.amount,
            s.sale_date
        FROM sales_view s
        JOIN products_view p
            ON s.product_id = p.product_id
        ORDER BY s.sale_id
        """
    )
    join_result.show()
    print_screen_marker(6, "Po JOIN sales_view z products_view")

    print("Wynik analityczny zapisywany do CSV")
    final_result = spark.sql(
        """
        SELECT
            p.category,
            s.region,
            COUNT(*) AS transaction_count,
            SUM(s.quantity) AS total_quantity,
            SUM(s.amount) AS total_amount,
            ROUND(AVG(s.amount), 2) AS average_amount
        FROM sales_view s
        JOIN products_view p
            ON s.product_id = p.product_id
        WHERE s.amount > 1000
        GROUP BY p.category, s.region
        ORDER BY p.category, s.region
        """
    )
    final_result.show()

    clean_output(QUERY_RESULT_FILE)
    final_result.toPandas().to_csv(QUERY_RESULT_FILE, index=False)

    print("Zapisano wynik zapytania do:", QUERY_RESULT_FILE)
    print("Kontrola odczytu zapisanego wyniku:")
    spark.read.csv(str(QUERY_RESULT_FILE), header=True, inferSchema=True).show()
    print_screen_marker(7, "Po zapisie wyniku do CSV i ponownym odczycie")

finally:
    spark.stop()
