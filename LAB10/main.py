from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQL_Parquet") \
    .master("local[*]") \
    .getOrCreate()

print("Spark działa!")

spark.stop()