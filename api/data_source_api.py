from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from kafka import KafkaProducer # New import
import json # New import
import logging # New import

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

# --- Kafka Producer Setup ---
# Use 'kafka:9092' as the hostname within the Docker network
producer = KafkaProducer(
    bootstrap_servers=['kafka:29092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'), # Serializes dicts to JSON bytes
    acks='all' # Ensures all in-sync replicas have received the message
)

@app.on_event("startup")
async def startup_event():
    logger.info("Data Source API starting up...")
    logger.info("Kafka producer initialized and connecting to kafka:9092...")
    # Optional: A small delay or check to ensure Kafka is ready.
    # In practice, depends_on in docker-compose helps, and producer retries internally.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Data Source API shutting down. Flushing and closing Kafka producer...")
    producer.flush() # Ensure all buffered messages are sent
    producer.close() # Close the producer connection
    logger.info("Kafka producer closed.")

@app.get("/")
async def read_root():
    return {"message": "Data Source API is running. Try /train or /test endpoints to send data to Kafka."}

@app.get("/health")
async def health_check():
    """
    A simple health check endpoint.
    Used by docker-compose's healthcheck.
    """
    return JSONResponse(content={"status": "healthy"}, status_code=200)

@app.get("/train")
async def send_train_data_to_kafka():
    """
    Reads Train.csv, sends each row as a message to the 'raw_train_data' Kafka topic.
    """
    file_path = os.path.join(DATA_PATH, 'Train.csv')
    if not os.path.exists(file_path):
        logger.error(f"Train.csv not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"Train.csv not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        messages_sent = 0
        logger.info(f"Starting to send {len(df)} train data records to 'raw_train_data' topic...")

        for index, row in df.iterrows():
            data_dict = row.to_dict()
            # Send the dictionary as a JSON message to Kafka
            producer.send('raw_train_data', data_dict)
            messages_sent += 1
            # In a very high-throughput scenario, you might send asynchronously
            # and check results later. For this example, simple send is fine.

        producer.flush() # Ensure all messages are sent before returning
        logger.info(f"Successfully sent {messages_sent} train data records to 'raw_train_data' topic.")
        return {"message": f"Successfully sent {messages_sent} train data records to Kafka", "topic": "raw_train_data"}
    except Exception as e:
        logger.error(f"Failed to read or send Train.csv data to Kafka: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read or send Train.csv data: {str(e)}")

@app.get("/test")
async def send_test_data_to_kafka():
    """
    Reads Test.csv, sends each row as a message to the 'raw_test_data' Kafka topic.
    """
    file_path = os.path.join(DATA_PATH, 'Test.csv')
    if not os.path.exists(file_path):
        logger.error(f"Test.csv not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"Test.csv not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        messages_sent = 0
        logger.info(f"Starting to send {len(df)} test data records to 'raw_test_data' topic...")

        for index, row in df.iterrows():
            data_dict = row.to_dict()
            # Send the dictionary as a JSON message to Kafka
            producer.send('raw_test_data', data_dict)
            messages_sent += 1

        producer.flush() # Ensure all messages are sent before returning
        logger.info(f"Successfully sent {messages_sent} test data records to 'raw_test_data' topic.")
        return {"message": f"Successfully sent {messages_sent} test data records to Kafka", "topic": "raw_test_data"}
    except Exception as e:
        logger.error(f"Failed to read or send Test.csv data to Kafka: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read or send Test.csv data: {str(e)}")