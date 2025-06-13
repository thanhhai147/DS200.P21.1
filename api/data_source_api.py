# api_producer/app.py (REVISED for Automatic, Mini-Batch Production from CSVs)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from kafka import KafkaProducer
import json
import logging
import time
import uuid
import threading
from datetime import datetime
import asyncio # For running async tasks if needed, though threading is used here

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Source API",
    description="API to serve Train and Test data from CSV files for streaming pipeline.",
    version="1.0.0"
)

# Define the base path for your data files inside the container
DATA_PATH = '/app/data' # This matches the COPY ../data /app/data/ in Dockerfile_API

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
RAW_TRAIN_TOPIC = "raw_train_data"
RAW_TEST_TOPIC = "raw_test_data"

# --- Global Producer Instance ---
producer_instance = None
producer_connected = False

# --- Global DataFrames and Pointers for Continuous Looping ---
train_df = None
test_df = None
train_data_idx = 0
test_data_idx = 0
data_production_active = True # Flag to control the background thread

# --- Kafka Producer Setup ---
def get_kafka_producer_with_retries(max_retries=15, delay_seconds=20):
    """
    Initializes and returns a KafkaProducer instance with retries.
    This helps handle Kafka not being fully ready immediately on container startup.
    """
    global producer_instance, producer_connected
    if producer_instance is None or not producer_connected:
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Attempt {retries + 1}/{max_retries}: Connecting Kafka Producer to {KAFKA_BOOTSTRAP_SERVERS}...")
                new_producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all'
                )
                new_producer.metrics() # Force a connection check
                producer_instance = new_producer
                producer_connected = True
                logger.info(f"Successfully connected Kafka Producer to {KAFKA_BOOTSTRAP_SERVERS}")
                return producer_instance
            except Exception as e:
                retries += 1
                logger.warning(f"Error connecting Kafka Producer: {e}. Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
        logger.error(f"Failed to connect Kafka Producer after {max_retries} retries. Data production will not start.")
        producer_connected = False
        return None
    return producer_instance

# --- Data Production Logic ---
def send_data_batch_to_kafka(df_source: pd.DataFrame, current_idx: int, topic: str, batch_size: int):
    """
    Sends a mini-batch of data from a DataFrame to a Kafka topic.
    Returns the next starting index for the DataFrame.
    """
    global producer_instance, producer_connected

    if not producer_connected:
        logger.error(f"Kafka Producer not connected. Cannot send data to topic {topic}.")
        return current_idx, 0

    records_sent = 0
    batch_end_idx = min(current_idx + batch_size, len(df_source))
    batch_data = df_source.iloc[current_idx:batch_end_idx]

    if batch_data.empty:
        return 0, 0 # Reset to beginning if no data, indicates end of current loop

    for _, row in batch_data.iterrows():
        data_dict = row.to_dict()
        data_dict['id'] = str(uuid.uuid4()) # Generate unique ID
        data_dict['date_time'] = datetime.now().isoformat() # Current timestamp

        try:
            producer_instance.send(topic, data_dict)
            records_sent += 1
            # logger.debug(f"Produced record {data_dict['id']} to {topic}")
        except Exception as e:
            logger.error(f"Failed to send message to {topic}: {e}. Attempting to re-initialize producer.")
            producer_instance = get_kafka_producer_with_retries() # Try to re-initialize
            if not producer_connected:
                logger.error("Could not re-initialize producer. Stopping data production.")
                data_production_active = False # Signal to stop production thread
                return current_idx, records_sent # Return current index, will exit loop
            
    producer_instance.flush() # Ensure all messages in this batch are sent
    logger.info(f"Sent {records_sent} records to Kafka topic '{topic}'.")

    next_idx = current_idx + records_sent
    if next_idx >= len(df_source):
        next_idx = 0 # Loop back to beginning of DataFrame

    return next_idx, records_sent

def continuous_data_production_task():
    """
    Background task to continuously read data from CSVs and send to Kafka.
    """
    global train_df, test_df, train_data_idx, test_data_idx, data_production_active

    logger.info("Starting continuous data production background task.")

    # Load data once at startup
    train_file_path = os.path.join(DATA_PATH, 'Train.csv')
    test_file_path = os.path.join(DATA_PATH, 'Test.csv')

    try:
        train_df = pd.read_csv(train_file_path)
        logger.info(f"Loaded {len(train_df)} records from Train.csv for training data production.")
    except Exception as e:
        logger.error(f"Failed to load Train.csv: {e}. Train data production will not occur.", exc_info=True)
        train_df = pd.DataFrame() # Empty DataFrame to prevent errors

    try:
        test_df = pd.read_csv(test_file_path)
        logger.info(f"Loaded {len(test_df)} records from Test.csv for test data production.")
    except Exception as e:
        logger.error(f"Failed to load Test.csv: {e}. Test data production will not occur.", exc_info=True)
        test_df = pd.DataFrame() # Empty DataFrame to prevent errors
    
    if train_df.empty and test_df.empty:
        logger.error("No data loaded from CSVs. Continuous production task terminating.")
        data_production_active = False
        return

    # Ensure Kafka producer is ready before starting the loop
    get_kafka_producer_with_retries()
    if not producer_connected:
        logger.error("Kafka producer is not connected. Aborting data production task.")
        return

    BATCH_SIZE = 16
    DELAY_BETWEEN_BATCHES_SECONDS = 20 # Adjust as needed

    while data_production_active:
        if not train_df.empty:
            train_data_idx, sent_count = send_data_batch_to_kafka(train_df, train_data_idx, RAW_TRAIN_TOPIC, BATCH_SIZE)
            if sent_count > 0:
                logger.info(f"Train data batch sent. Next train index: {train_data_idx}")
            else:
                logger.debug("No train data sent in this batch.")
        else:
            logger.debug("Train data DataFrame is empty.")

        if not test_df.empty:
            test_data_idx, sent_count = send_data_batch_to_kafka(test_df, test_data_idx, RAW_TEST_TOPIC, BATCH_SIZE)
            if sent_count > 0:
                logger.info(f"Test data batch sent. Next test index: {test_data_idx}")
            else:
                logger.debug("No test data sent in this batch.")
        else:
            logger.debug("Test data DataFrame is empty.")

        time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)
    
    logger.info("Data production background task stopped.")


# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Data Source API starting up...")
    # Start the continuous data production in a separate thread
    production_thread = threading.Thread(target=continuous_data_production_task)
    production_thread.daemon = True # Allow main program to exit if Flask app shuts down
    production_thread.start()
    logger.info("Continuous data production task started in background.")

@app.on_event("shutdown")
async def shutdown_event():
    global data_production_active
    logger.info("Data Source API shutting down. Signaling production task to stop...")
    data_production_active = False # Signal the background thread to stop
    # Give it a moment to finish current work, then force flush/close
    time.sleep(2) 
    if producer_instance:
        logger.info("Flushing and closing Kafka producer...")
        producer_instance.flush()
        producer_instance.close()
        logger.info("Kafka producer closed.")
    logger.info("Data Source API shutdown complete.")

# --- FastAPI Routes ---
@app.get("/")
async def read_root():
    return {"message": "Data Source API is running. Data production is automatic."}

@app.get("/health")
async def health_check():
    """
    A simple health check endpoint.
    Used by docker-compose's healthcheck.
    """
    return JSONResponse(content={"status": "healthy", "producer_connected": producer_connected}, status_code=200)

# Removed /train and /test manual trigger endpoints as production is now automatic.