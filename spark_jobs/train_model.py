# spark_jobs/train_model.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, IndexToString
from pyspark.ml import PipelineModel
import logging
import os
import shutil

# --- 1. Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Spark Session Configuration ---
SPARK_PACKAGES = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
spark = SparkSession.builder \
    .appName("SentimentModelTraining") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .config("spark.cassandra.connection.host", "cassandra") \
    .getOrCreate()

# Setting log level for Spark's internal (Java/Scala) logs
spark.sparkContext.setLogLevel("WARN")
logger.info("Spark Session for Model Training created.")

# --- 3. Configuration Variables ---
CASSANDRA_KEYSPACE = "my_project_keyspace"
CASSANDRA_TRAIN_TABLE = "raw_train_data"
MODEL_SAVE_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model" # Path inside the spark-master container

# --- 4. Data Loading from Cassandra ---
logger.info(f"Attempting to read train data from Cassandra: {CASSANDRA_KEYSPACE}.{CASSANDRA_TRAIN_TABLE}")
try:
    train_df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=CASSANDRA_TRAIN_TABLE, keyspace=CASSANDRA_KEYSPACE) \
        .load()

    num_rows_full = train_df.count()
    logger.info(f"Successfully loaded {num_rows_full} rows from Cassandra for training.")
    logger.info("Schema of loaded training data:")
    train_df.printSchema()
    logger.info("First 5 rows of loaded training data (full dataset):")
    train_df.show(5, truncate=False)

    # --- 5. Data Sampling for Testing (Recommended for faster iteration) ---
    SAMPLE_FRACTION = 0.1 # 10% for faster testing
    RANDOM_SEED = 42      # For reproducibility of the sample
    train_df = train_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=RANDOM_SEED)

    num_rows_sample = train_df.count()
    logger.info(f"Using a sample of {num_rows_sample} rows ({SAMPLE_FRACTION*100}%) for training.")
    logger.info("First 5 rows of SAMPLED training data:")
    train_df.show(5, truncate=False)

except Exception as e:
    logger.error(f"Failed to read data from Cassandra: {e}", exc_info=True)
    raise e # Re-raise to ensure the job fails visibly

# --- 6. UDF to Normalize Labels ---
@udf(StringType())
def normalize_and_concat_labels(label_str):
    if label_str is None:
        return None
    individual_labels = [s.strip() for s in label_str.split(';') if s.strip()]
    if not individual_labels:
        return None
    return ";".join(sorted(list(set(individual_labels))))

logger.info("Applying UDF to normalize and concatenate labels...")
train_df = train_df.withColumn("processed_label", normalize_and_concat_labels(col("label")))
logger.info("Labels normalization complete.")
logger.info("Comparison of original vs. processed labels (sampled data):")
train_df.select("label", "processed_label").show(5, truncate=False)

# --- 7. ML Pipeline Stages Definition ---
logger.info("Defining ML Pipeline stages...")

tokenizer = Tokenizer(inputCol="comment", outputCol="words")
logger.info("Tokenizer stage defined.")

hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features") # IDF refines the features
logger.info("HashingTF and IDF featurization stages defined.")

logger.info("Fitting StringIndexer for 'processed_label'...")
label_indexer = StringIndexer(inputCol="processed_label", outputCol="indexed_label", handleInvalid="skip").fit(train_df)
logger.info("StringIndexer fitted. Unique labels found:")
for i, label in enumerate(label_indexer.labels):
    logger.info(f"   [{i}]: {label}")

# Logistic Regression model: maxIter=10 is good for initial testing.
lr = LogisticRegression(
    featuresCol="features",
    labelCol="indexed_label",
    maxIter=10,
    probabilityCol="probability" # <-- ADDED: Explicitly request probability output
)
logger.info("LogisticRegression model defined with probability output.")

# Converts numerical predictions back to original string labels.
label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_composite_label", # <--- Confirmed this is the output column name
    labels=label_indexer.labels
)
logger.info("IndexToString converter defined.")

# --- 8. Build and Train the ML Pipeline ---
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, lr, label_converter])
logger.info("ML Pipeline assembled.")
logger.info("Starting model training (pipeline.fit()). This may take a while depending on data size and resources...")

# --- THE ACTUAL TRAINING CALL ---
trained_model = pipeline.fit(train_df)
logger.info("Model training complete. PipelineModel fitted successfully.")

# --- 9. Save the Trained Model for Prediction/Inference ---
# We include the stages that generate all desired output columns for prediction.
# Original pipeline stages indices:
# 0: tokenizer, 1: hashing_tf, 2: idf, 3: label_indexer, 4: lr (LogisticRegressionModel), 5: label_converter (IndexToString)
prediction_stages = [
    trained_model.stages[0], # tokenizer (fitted)
    trained_model.stages[1], # hashing_tf (fitted)
    trained_model.stages[2], # idf (fitted)
    trained_model.stages[4], # fitted LogisticRegressionModel - generates prediction, rawPrediction, probability
    trained_model.stages[5]  # fitted IndexToString - generates predicted_composite_label from 'prediction'
]

# Create a new PipelineModel with only these prediction-relevant stages
prediction_pipeline_model = PipelineModel(stages=prediction_stages)
logger.info("Created a new PipelineModel containing prediction-relevant stages (Tokenizer, HashingTF, IDF, LR Model, IndexToString).")

# --- Save this new prediction_pipeline_model ---
logger.info(f"Attempting to save the prediction model to {MODEL_SAVE_PATH}...")
try:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    prediction_pipeline_model.write().overwrite().save(MODEL_SAVE_PATH)
    logger.info(f"Prediction model saved successfully to {MODEL_SAVE_PATH}")
except Exception as e:
    logger.error(f"Failed to save prediction model to {MODEL_SAVE_PATH}: {e}", exc_info=True)
    raise e

# --- 10. Stop Spark Session ---
spark.stop()
logger.info("Spark Session stopped.")