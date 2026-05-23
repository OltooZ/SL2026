import os
import sys

from pyspark.sql import SparkSession

# wymuszenie poprawnego interpretera Pythona
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("TestSpark") \
    .master("local[*]") \
    .config("spark.python.worker.reuse", "false") \
    .getOrCreate()

data = [
    ("Patryk", 22),
    ("Adam", 25),
    ("Kasia", 21)
]

df = spark.createDataFrame(data, ["Imie", "Wiek"])

df.show()

spark.stop()