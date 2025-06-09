# spark_jobs/process_test_data.py (UPDATED for dashboard output)
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import from_json, col, current_timestamp, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.ml import PipelineModel
import logging
import uuid
import os # Added for path operations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark Session
SPARK_PACKAGES = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"

spark = SparkSession.builder \
    .appName("KafkaTestDataProcessorAndPredictor") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
logger.info("Spark Session for Test Data Processor created.")

# --- Define Schema for Incoming Test Data from Kafka ---
test_data_kafka_schema = StructType([
    StructField("comment", StringType(), True),
    StructField("n_star", IntegerType(), True),
    StructField("date_time", StringType(), True),
    StructField("label", StringType(), True), # Crucial for carrying true label through for dashboard
])

# UDF to generate a UUID
generate_uuid_udf = udf(lambda: str(uuid.uuid4()), StringType())

# --- Load the Pre-Trained ML Model ---
MODEL_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model"
try:
    model = PipelineModel.load(MODEL_PATH)
    logger.info(f"Machine Learning Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load ML model from {MODEL_PATH}: {e}")
    raise e

# Read data from Kafka (raw_test_data topic)
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "raw_test_data") \
    .option("startingOffsets", "latest") \
    .load()

logger.info("Reading stream from raw_test_data Kafka topic...")

# Parse the JSON 'value' from Kafka, add UUID, parse 'date_time'
parsed_test_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), test_data_kafka_schema).alias("data")) \
    .select("data.*") \
    .withColumn("id", generate_uuid_udf()) \
    .withColumn("date_time", col("date_time").cast(TimestampType())) \
    .withColumn("processing_timestamp", current_timestamp())

# --- Make Predictions using the Loaded Model ---
# The model.transform method will apply all pipeline stages
# (tokenization, featurization, prediction, probability generation, and IndexToString for string label)
predictions_df = model.transform(parsed_test_df)

# --- DEBUGGING AID: Print schema to confirm columns from model (very useful) ---
logger.info("Schema of predictions_df after model.transform():")
predictions_df.printSchema()
# --- END DEBUGGING AID ---

# Select and format data for output to predicted_test_data Kafka topic
output_df = predictions_df.select(
    "id",
    "comment",
    "n_star",
    "date_time",
    "label", # Original true label from incoming Kafka data for comparison on dashboard
    F.col("prediction").alias("predicted_label_index"), # The raw prediction index (0 or 1)
    F.col("predicted_composite_label").alias("predicted_sentiment_string"), # Human-readable predicted label (e.g., "positive", "negative")
    "rawPrediction", # rawPrediction vector (e.g., [score_neg, score_pos])
    # F.col("probability.values")[1].alias("positive_probability") # Probability of the positive class (from the 'values' array in the probability vector)
)

# Convert the DataFrame to JSON strings for Kafka output
# All selected columns will be included in the JSON message.
output_kafka_df = output_df.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value")

# Define output Kafka topic for predictions
PREDICTED_KAFKA_TOPIC = "predicted_test_data"
logger.info(f"Writing predictions to Kafka topic: {PREDICTED_KAFKA_TOPIC}")

# Write predictions to the 'predicted_test_data' Kafka topic
query_predictions = output_kafka_df \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("topic", PREDICTED_KAFKA_TOPIC) \
    .option("checkpointLocation", os.path.join("/tmp/spark/checkpoints", PREDICTED_KAFKA_TOPIC)) \
    .trigger(processingTime="10 seconds") \
    .start()

logger.info("Spark Structured Streaming query for predictions started.")

# Await termination of the query
query_predictions.awaitTermination()

# Stop Spark Session
spark.stop()
logger.info("Spark Session stopped.")