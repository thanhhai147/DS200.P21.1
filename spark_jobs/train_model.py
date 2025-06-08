# spark_jobs/train_model.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType 
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, IndexToString
from pyspark.ml import PipelineModel
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark Session
SPARK_PACKAGES = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
spark = SparkSession.builder \
    .appName("SentimentModelTraining") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config('spark.cassandra.connection.port', '9042') \
    .config('spark.cassandra.output.consistency.level','ONE') \
    .master('local[2]') \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN") # Keep Spark's internal logs concise
logger.info("Spark Session for Model Training created.")

# Define Cassandra keyspace and table for train data
CASSANDRA_KEYSPACE = "my_project_keyspace"
CASSANDRA_TRAIN_TABLE = "raw_train_data"

# Path to save the trained model
MODEL_SAVE_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model"

logger.info(f"Attempting to read train data from Cassandra: {CASSANDRA_KEYSPACE}.{CASSANDRA_TRAIN_TABLE}")

# Read training data from Cassandra
try:
    train_df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=CASSANDRA_TRAIN_TABLE, keyspace=CASSANDRA_KEYSPACE) \
        .load()
    num_rows = train_df.count() # Trigger action to count rows
    logger.info(f"Successfully loaded {num_rows} rows from Cassandra for training.")
    logger.info("Schema of loaded training data:")
    train_df.printSchema()
    logger.info("First 5 rows of loaded training data:")
    train_df.show(5, truncate=False) # Show more of the label string
except Exception as e:
    logger.error(f"Failed to read data from Cassandra: {e}", exc_info=True)
    raise e

# --- UDF to Normalize Labels ---
@udf(StringType())
def normalize_and_concat_labels(label_str):
    if label_str is None:
        return None
    individual_labels = [s.strip() for s in label_str.split(';') if s.strip()]
    if not individual_labels:
        return None
    return ";".join(sorted(list(set(individual_labels))))

logger.info("Applying UDF to normalize and concatenate labels...")
# Apply the UDF to create a new label column for training
train_df = train_df.withColumn("processed_label", normalize_and_concat_labels(col("label")))
logger.info("Original labels processed into 'processed_label' for training.")
logger.info("Comparison of original vs. processed labels:")
train_df.select("label", "processed_label").show(5, truncate=False)

# --- NEW: Sample 10% of the data for testing ---
SAMPLE_FRACTION = 0.1 # 10%
RANDOM_SEED = 42    # For reproducibility
train_df = train_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=RANDOM_SEED)
num_rows_sample = train_df.count() # Trigger action to count sampled rows
logger.info(f"Using a sample of {num_rows_sample} rows ({SAMPLE_FRACTION*100}%) for training.")
train_df.show(5, truncate=False) # Show sample data

# --- ML Pipeline Stages ---
logger.info("Defining ML Pipeline stages...")

tokenizer = Tokenizer(inputCol="comment", outputCol="words")
logger.info("Tokenizer stage defined.")

hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
logger.info("HashingTF and IDF featurization stages defined.")

logger.info("Fitting StringIndexer for 'processed_label'...")
# 3. Index the 'processed_label' column
label_indexer = StringIndexer(inputCol="processed_label", outputCol="indexed_label", handleInvalid="skip").fit(train_df)
logger.info("StringIndexer fitted. Unique labels found:")
for i, label in enumerate(label_indexer.labels):
    logger.info(f"  [{i}]: {label}")

lr = LogisticRegression(featuresCol="features", labelCol="indexed_label", maxIter=10)
logger.info("LogisticRegression model defined.")

label_converter = IndexToString(inputCol="prediction", outputCol="predicted_composite_label",
                                labels=label_indexer.labels)
logger.info("IndexToString converter defined.")

# --- Build the ML Pipeline ---
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, lr, label_converter])
logger.info("ML Pipeline assembled.")

logger.info("Starting model training (pipeline.fit()). This may take a while depending on data size and resources...")
# Train the model
trained_model = pipeline.fit(train_df)
logger.info("Model training complete. PipelineModel fitted.")

# --- Save the Trained Model ---
logger.info(f"Attempting to save the trained model to {MODEL_SAVE_PATH}...")
try:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    if os.path.exists(MODEL_SAVE_PATH):
        logger.info(f"Existing model directory found at {MODEL_SAVE_PATH}. Deleting it before saving new model.")
        shutil.rmtree(MODEL_SAVE_PATH)
    
    trained_model.write().save(MODEL_SAVE_PATH)
    logger.info(f"Trained model saved successfully to {MODEL_SAVE_PATH}")
except Exception as e:
    logger.error(f"Failed to save model to {MODEL_SAVE_PATH}: {e}", exc_info=True)
    raise e

spark.stop()
logger.info("Spark Session stopped.")