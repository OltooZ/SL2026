from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("test") \
    .master("local[*]") \
    .getOrCreate()

data = [(1, "A"), (2, "B")]
df = spark.createDataFrame(data, ["id", "name"])

df.show()

spark.stop()