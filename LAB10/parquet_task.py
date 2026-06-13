import os
import sys

from pyspark.sql import SparkSession

# Interpreter Pythona dla Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("SparkSQL_Parquet") \
    .master("local[1]") \
    .config("spark.python.worker.reuse", "false") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.file.impl",
            "org.apache.hadoop.fs.local.LocalFs") \
    .getOrCreate()

# WYŁĄCZENIE native Hadoop
spark.sparkContext._jsc.hadoopConfiguration().set(
    "io.native.lib.available", "false"
)

# Dane testowe
data = [
    (1, "Laptop", 3200, "Elektronika"),
    (2, "Telefon", 2400, "Elektronika"),
    (3, "Biurko", 900, "Meble"),
    (4, "Krzesło", 450, "Meble"),
    (5, "Monitor", 1200, "Elektronika")
]

columns = ["id", "produkt", "cena", "kategoria"]

df = spark.createDataFrame(data, columns)

print("Dane wejściowe:")
df.show()

# Folder lokalny
path = "C:/tmp/produkty_parquet"

# zapis
df.write.mode("overwrite").parquet(path)

print("Plik Parquet został utworzony!")

# odczyt
df_parquet = spark.read.parquet(path)

print("Dane z pliku Parquet:")
df_parquet.show()

print("Schemat danych:")
df_parquet.printSchema()

spark.stop()