# spark_jobs/process_train_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import logging
import uuid # For generating UUIDs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark Session
# These packages are crucial for connecting to Kafka and Cassandra
# Ensure the Scala version (e.g., _2.12) and Spark version (e.g., 3.5.0) match your setup
SPARK_PACKAGES = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"

spark = SparkSession.builder \
    .appName("KafkaTrainDataToCassandra") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .config("spark.cassandra.connection.host", "cassandra") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN") # Set Spark logging level to WARN to reduce verbosity
logger.info("Spark Session for Train Data Processor created.")

# --- Define Schema for Incoming Kafka Messages (from your API) ---
# This schema MUST match the JSON structure your API sends from Train.csv
# 'date_time' is a StringType because it comes as a string from your CSV/API
train_data_kafka_schema = StructType([
    StructField("comment", StringType(), True),
    StructField("n_star", IntegerType(), True),
    StructField("date_time", StringType(), True), # API sends this as a string from CSV
    StructField("label", StringType(), True),
])

# UDF (User-Defined Function) to generate a UUID for the primary key
# This will create a unique ID for each record for Cassandra
generate_uuid_udf = udf(lambda: str(uuid.uuid4()), StringType())

# Read data from Kafka (raw_train_data topic)
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "raw_train_data") \
    .option("startingOffsets", "latest") \
    .load()

logger.info("Reading stream from raw_train_data Kafka topic...")

# Parse the JSON 'value' from Kafka, add a UUID, parse 'date_time', and add a processing timestamp
processed_train_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), train_data_kafka_schema).alias("data")) \
    .select("data.*") \
    .withColumn("id", generate_uuid_udf()) \
    .withColumn("date_time", col("date_time").cast(TimestampType())) \
    .withColumn("processing_timestamp", current_timestamp()) # Add a timestamp when Spark processed the record

# Select and reorder columns to match the Cassandra table exactly
# Ensure these column names and order match your 'raw_train_data' table in Cassandra
final_train_df_to_cassandra = processed_train_df.select(
    col("id"),
    col("comment"),
    col("n_star"),
    col("date_time"),
    col("label"),
    col("processing_timestamp")
)

# Define Cassandra keyspace and table for train data
CASSANDRA_KEYSPACE = "my_project_keyspace"
CASSANDRA_TRAIN_TABLE = "raw_train_data"

logger.info(f"Writing processed train data to Cassandra table: {CASSANDRA_KEYSPACE}.{CASSANDRA_TRAIN_TABLE}")

# Write processed train data to Cassandra
# 'checkpointLocation' is essential for fault tolerance in Structured Streaming
# 'foreachBatch' is used for Cassandra writes to ensure proper semantics
query_train = final_train_df_to_cassandra \
    .writeStream \
    .option("checkpointLocation", "/tmp/spark/checkpoints/train_data_to_cassandra") \
    .foreachBatch(lambda df, epoch_id: df.write \
                                       .format("org.apache.spark.sql.cassandra") \
                                       .options(table=CASSANDRA_TRAIN_TABLE, keyspace=CASSANDRA_KEYSPACE) \
                                       .mode("append") \
                                       .save()) \
    .trigger(processingTime="5 seconds") \
    .start() # Start the streaming query

logger.info("Spark Structured Streaming query for train data started.")

# Await termination of the query. This keeps the Spark application running continuously.
query_train.awaitTermination()