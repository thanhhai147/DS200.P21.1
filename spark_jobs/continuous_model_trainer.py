# spark_jobs/continuous_model_trainer.py (with Retry Logic)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, explode, split, when, collect_list, array_contains, lit, expr
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, IndexToString
import logging
import os
import shutil
import fcntl  
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Spark Session Configuration ---
SPARK_PACKAGES = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
spark = SparkSession.builder \
    .appName("ContinuousMultiClassTrainer") \
    .config("spark.jars.packages", SPARK_PACKAGES) \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.local.dir", "/opt/spark_temp_data") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
logger.info("Spark Session for Continuous Multi-Class Model Trainer created.")

# --- Configuration Variables ---
CASSANDRA_KEYSPACE = "bigdata_keyspace"
CASSANDRA_TRAIN_TABLE = "raw_train_data"
MODEL_SAVE_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model"
MODEL_TEMP_SAVE_PATH = "/opt/bitnami/spark/jobs/model/sentiment_model_temp"
LOCK_FILE_NAME = "training.lock"
LOCK_FILE_PATH = os.path.join(MODEL_SAVE_PATH, LOCK_FILE_NAME)

aspect_cols = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE', 'OTHERS']

@contextmanager
def file_lock(lock_file_path):
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(lock_file_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # Exclusive lock
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

# --- Main Training Logic ---
def train_and_save_model(batch_df, batch_id):
    with file_lock(LOCK_FILE_PATH):

        logger.info(f"Batch {batch_id}: Triggering model retraining cycle.")

        # 1. Read the complete training dataset from Cassandra
        train_df = None
        
        logger.info(f"Batch {batch_id}: Reading training data from Cassandra.")
        train_df = spark.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table=CASSANDRA_TRAIN_TABLE, keyspace=CASSANDRA_KEYSPACE) \
            .load()
        # Force an action to ensure data is read and potential errors occur here
        if train_df.head(1): # Try to fetch a single row to trigger the read
            logger.info(f"Batch {batch_id}: Successfully read data from Cassandra.")

        if train_df is None: # If read failed and loop completed without break
                logger.error(f"Batch {batch_id}: Training data DataFrame is None after retries. Skipping retraining.")
                return

        num_records = train_df.count()
        if num_records == 0:
            logger.warning(f"Batch {batch_id}: No training data in Cassandra after read. Skipping retraining.")
            return

        logger.info(f"Batch {batch_id}: Starting training with {num_records} records.")

        # 2. Pre-process Data: Create a single normalized label column
        # Bước 1: Tách tất cả các label từ chuỗi label (dạng: "{BATTERY#Positive}{CAMERA#Negative}")
        df_labels = train_df.withColumn(
            "LabelList",
            split(regexp_replace(col("label"), r"[{}]", ""), r";")
        ).withColumn(
            "LabelList",
            expr("filter(LabelList, x -> x != '')")  # loại bỏ item rỗng
        )

        # Bước 2: Explode LabelList thành từng dòng để xử lý
        exploded_df = df_labels.withColumn("label_entry", explode(col("LabelList")))

        # Bước 3: Tách thành aspect và sentiment
        split_df = exploded_df.withColumn("aspect", split(col("label_entry"), "#").getItem(0)) \
                            .withColumn("sentiment", split(col("label_entry"), "#").getItem(1))
        
        # Bước 4: Chuyển sentiment thành số
        sentiment_map = {"Negative": 1, "Neutral": 2, "Positive": 3}
        sentiment_expr = F.create_map([lit(kv) for kv in sum(sentiment_map.items(), ())])
        split_df = split_df.withColumn("sentiment_value", sentiment_expr.getItem(col("sentiment")))

        # Bước 5: Pivot về dạng wide (mỗi aspect là 1 cột)
        pivot_df = split_df.filter(col("aspect") != "OTHERS") \
            .groupBy("comment") \
            .pivot("aspect") \
            .agg(F.first("sentiment_value"))
        
        # Bước 6: Xử lý riêng cột OTHERS (nếu xuất hiện thì là 1, ngược lại 0)
        others_df = split_df.groupBy("comment") \
            .agg(F.max(when(col("aspect") == "OTHERS", 1).otherwise(0)).alias("OTHERS"))

        # Bước 7: Gộp lại với các cột còn lại
        comment_df = split_df.select("comment").distinct()

        processed_df = comment_df \
            .join(pivot_df, on="comment", how="left") \
            .join(others_df, on="comment", how="left")

        # Điền 0 vào các ô null (tức là không có aspect đó)
        for aspect in aspect_cols:
            processed_df = processed_df.withColumn(aspect, F.coalesce(col(aspect), lit(0)))

        if processed_df.count() == 0:
            logger.warning(f"Batch {batch_id}: No valid processed labels after normalization. Skipping retraining.")
            return

        for aspect in aspect_cols:
            
            if aspect not in processed_df.columns:
                continue
            # 3. Define the ML Pipeline for a Single Multi-Class Model
            tokenizer = Tokenizer(inputCol="comment", outputCol="words")
            hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=3000)
            idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
            lr = LogisticRegression(
                featuresCol="tfidf_features",
                labelCol=aspect,
                maxIter=100,
                predictionCol="prediction",
                family="multinomial"
            )
            
            pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, lr])

            # 4. Fit the pipeline on the training data
            logger.info(f"Batch {batch_id}: Fitting the {aspect} pipeline...")
            pipeline_model = pipeline.fit(processed_df)
            logger.info(f"Batch {batch_id}: {aspect} pipeline fitting complete.")

            # 5. Save the Model Atomically
            ASPECT_MODEL_TEMP_SAVE_PATH = os.path.join(MODEL_TEMP_SAVE_PATH, aspect)
            ASPECT_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, aspect)
            logger.info(f"Batch {batch_id}: Saving {aspect} model to temporary location: {ASPECT_MODEL_TEMP_SAVE_PATH}")
                    
            # Clean up previous temp dir if it exists
            if os.path.exists(ASPECT_MODEL_TEMP_SAVE_PATH):
                shutil.rmtree(ASPECT_MODEL_TEMP_SAVE_PATH)
            
            # Ensure parent directorys exists for temp save
            os.makedirs(os.path.dirname(ASPECT_MODEL_TEMP_SAVE_PATH), exist_ok=True)
            
            pipeline_model.write().overwrite().save(ASPECT_MODEL_TEMP_SAVE_PATH)
            logger.info(f"Batch {batch_id}: {aspect} model saved to temporary location successfully.")
            
            # Atomically replace the old model with the new one.
            logger.info(f"Batch {batch_id}: Atomically replacing old model with new model.")
            # Clean up old model if it exists
            if os.path.exists(ASPECT_MODEL_SAVE_PATH):
                shutil.rmtree(ASPECT_MODEL_SAVE_PATH)
            os.rename(ASPECT_MODEL_TEMP_SAVE_PATH, ASPECT_MODEL_SAVE_PATH)
            logger.info(f"Batch {batch_id}: New {aspect} model saved successfully to {ASPECT_MODEL_SAVE_PATH}")


# --- Streaming Query Definition ---
trigger_stream = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 1) \
    .load()

logger.info("Starting continuous model trainer. A new training cycle will start every 5 minutes.")
query_trainer = trigger_stream.writeStream \
    .foreachBatch(train_and_save_model) \
    .option("checkpointLocation", "/tmp/spark/checkpoints/continuous_multiclass_trainer_checkpoint") \
    .trigger(processingTime="300 seconds") \
    .start()

logger.info("Continuous Multi-Class Trainer streaming query has started.")
query_trainer.awaitTermination()

spark.stop()
logger.info("Spark Session stopped.")