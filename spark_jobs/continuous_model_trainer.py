# spark_jobs/continuous_model_trainer.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, TimestampType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, IndexToString
from pyspark.ml import PipelineModel
import logging
import os
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Spark Session Configuration ---
SPARK_PACKAGES = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
spark = SparkSession.builder \
    .appName("ContinuousSentimentModelTrainer") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .config("spark.cassandra.connection.host", "cassandra") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
logger.info("Spark Session for Continuous Model Trainer created.")

# --- Configuration Variables ---
CASSANDRA_KEYSPACE = "my_project_keyspace"
CASSANDRA_TRAIN_TABLE = "raw_train_data"
MODEL_SAVE_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model" # Shared volume path

# --- UDF to Normalize Labels (same as in train_model.py) ---
@udf(StringType())
def normalize_and_concat_labels(label_str):
    if label_str is None:
        return None
    individual_labels = [s.strip() for s in label_str.split(';') if s.strip()]
    if not individual_labels:
        return None
    return ";".join(sorted(list(set(individual_labels))))

# --- Main Training Logic ---
def train_and_save_model(batch_df, batch_id):
    """
    This function is called for each micro-batch.
    It will read ALL available training data from Cassandra and retrain the model.
    """
    logger.info(f"Batch {batch_id}: Triggering model retraining.")

    try:
        # Read ALL available training data from Cassandra for retraining
        # This will include all historical data + newly ingested data since last training
        train_df = spark.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table=CASSANDRA_TRAIN_TABLE, keyspace=CASSANDRA_KEYSPACE) \
            .load()
        
        # --- Data Sampling for Faster Testing (Optional, but good for frequent retraining) ---
        # Consider a smaller sample if retraining is very frequent and dataset is large
        SAMPLE_FRACTION = 1.0 # Use full dataset for production, 0.1 for rapid testing
        if SAMPLE_FRACTION < 1.0:
            train_df = train_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)

        num_records_for_training = train_df.count()
        if num_records_for_training == 0:
            logger.warning(f"Batch {batch_id}: No training data available in Cassandra. Skipping retraining.")
            return

        logger.info(f"Batch {batch_id}: Training model using {num_records_for_training} records from Cassandra.")

        # Apply UDF to normalize and concatenate labels
        train_df = train_df.withColumn("processed_label", normalize_and_concat_labels(col("label")))

        # Define ML Pipeline stages
        tokenizer = Tokenizer(inputCol="comment", outputCol="words")
        hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000) # numFeatures=1000 (reduced for faster training)
        idf = IDF(inputCol="raw_features", outputCol="features")

        # StringIndexer: Needs to be fitted on the current training data
        label_indexer = StringIndexer(inputCol="processed_label", outputCol="indexed_label", handleInvalid="skip")
        
        # Logistic Regression model
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="indexed_label",
            maxIter=10, # Keep maxIter low for frequent updates
            probabilityCol="probability" # Explicitly request probability output
        )

        # IndexToString: Converts numerical predictions back to original string labels.
        label_converter = IndexToString(
            inputCol="prediction",
            outputCol="predicted_composite_label",
            labels=label_indexer.fit(train_df).labels # Fit StringIndexer here to get labels for converter
        )

        # Build and Train the ML Pipeline
        # NOTE: StringIndexer is fitted inside the pipeline.fit() call if it's included as an Estimator
        # However, for label_converter to correctly map, we need the fitted label_indexer's labels.
        # This is why it's fitted separately here.
        # Alternatively, ensure the full pipeline includes the Estimator for StringIndexer.
        # For simplicity and to ensure `label_converter` has the correct `labels` set during `fit`,
        # it's common to fit the `StringIndexer` first, then pass its fitted labels to `IndexToString`.
        # Corrected pipeline definition to ensure labels are passed correctly:
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, lr, label_converter])
        
        logger.info(f"Batch {batch_id}: Starting model training (pipeline.fit()).")
        trained_model = pipeline.fit(train_df)
        logger.info(f"Batch {batch_id}: Model training complete. PipelineModel fitted.")

        # Extract only the prediction-relevant stages for saving
        # These indices depend on the order in your pipeline definition.
        # (tokenizer, hashing_tf, idf, label_indexer, lr, label_converter)
        # Note: label_indexer (index 3) is a Transformer after fit.
        # We need its 'labels' for IndexToString, which is part of the saved model.
        prediction_stages = [
            trained_model.stages[0], # tokenizer
            trained_model.stages[1], # hashing_tf
            trained_model.stages[2], # idf
            trained_model.stages[4], # lr (LogisticRegressionModel)
            trained_model.stages[5]  # label_converter (IndexToString)
        ]
        prediction_pipeline_model = PipelineModel(stages=prediction_stages)

        # Save the new model (overwriting the old one)
        logger.info(f"Batch {batch_id}: Attempting to save new prediction model to {MODEL_SAVE_PATH}...")
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            prediction_pipeline_model.write().overwrite().save(MODEL_SAVE_PATH)
            logger.info(f"Batch {batch_id}: New prediction model saved successfully to {MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"Batch {batch_id}: Failed to save new model to {MODEL_SAVE_PATH}: {e}", exc_info=True)
            # Re-raise if saving is critical, otherwise log and continue
            raise e

    except Exception as e:
        logger.error(f"Batch {batch_id}: Error during model retraining: {e}", exc_info=True)
        # Continue to the next batch even if this one fails

# A dummy stream to trigger the foreachBatch.
# It doesn't read actual data, but acts as a trigger mechanism.
# We'll use a rate source or a file source that is always available.
# A rate source is best for demonstrating periodic triggers.
trigger_stream = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 1) \
    .load()

logger.info(f"Starting continuous model trainer. Model will retrain every 60 seconds.")
# Use a processingTime trigger to retrain periodically
query_trainer = trigger_stream.writeStream \
    .foreachBatch(train_and_save_model) \
    .option("checkpointLocation", "/tmp/spark/checkpoints/continuous_trainer_checkpoint") \
    .trigger(processingTime="60 seconds") \
    .start()

logger.info("Continuous Model Trainer Streaming Query started.")

# Await termination to keep the Spark application running
query_trainer.awaitTermination()

spark.stop()
logger.info("Spark Session stopped.")