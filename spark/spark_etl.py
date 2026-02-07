import os
import pathlib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import NumericType

# Load environment variables
from dotenv import load_dotenv

env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# initialize directory paths
BASE_DIR = pathlib.Path(__file__).parent.parent
CSV_DIR = os.path.join(BASE_DIR, "data", "")

spark = (
    SparkSession.builder.appName("Fintech-ETL")
    .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
    .getOrCreate()
)

df = spark.read.csv(
    CSV_DIR + "Fraud Detection Dataset.csv", header=True, inferSchema=True
)

# Data cleaning with inserting median for null values
numeric_columns = [
    field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)
]

median_values = {}
for column in numeric_columns:
    median = df.approxQuantile(col=column, probabilities=[0.5], relativeError=0.01)[0]
    median_values[column] = median

# Fill null values with median
df = df.na.fill(median_values)

# write data
df.write.format("jdbc") \
    .option("url", os.getenv("DB_URL")) \
    .option("dbtable", os.getenv("DB_TABLE")) \
    .option("user", os.getenv("DB_USER")) \
    .option("password", os.getenv("DB_PASSWORD")) \
    .option("driver", "org.postgresql.Driver") \
    .mode("append") \
    .save()
