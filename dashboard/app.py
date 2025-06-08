import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import time

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Spark Model Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Real-time Spark Model Predictions")
st.write("Monitoring predictions from the Spark inference pipeline.")

# --- Kafka Configuration ---
# Use the service name 'kafka' and internal port 29092 as defined in docker-compose.yml
KAFKA_BOOTSTRAP_SERVERS = 'kafka:29092'
PREDICTED_TOPIC = 'predicted_test_data' # This topic will receive predictions from Spark

# --- Kafka Consumer Initialization ---
@st.cache_resource
def get_kafka_consumer():
    """
    Initializes and returns a KafkaConsumer instance.
    @st.cache_resource ensures the consumer is created only once.
    """
    try:
        consumer = KafkaConsumer(
            PREDICTED_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='latest', # Start consuming from the latest offset
            enable_auto_commit=True,
            group_id='streamlit-dashboard-group', # Consumer group ID
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        st.success(f"Successfully connected to Kafka topic: {PREDICTED_TOPIC}")
        return consumer
    except Exception as e:
        st.error(f"Error connecting to Kafka: {e}")
        st.warning(f"Please ensure Kafka is running and topic '{PREDICTED_TOPIC}' exists.")
        return None

consumer = get_kafka_consumer()

# --- Display Data ---
if consumer:
    st.header("Latest Predictions:")
    # Placeholder for the data table
    data_container = st.empty()
    
    # Initialize an empty list to store predictions
    predictions_list = []

    # Simple loop to continuously read from Kafka
    while True:
        # poll for new messages, timeout after 100ms
        messages = consumer.poll(timeout_ms=100)
        if messages:
            for topic_partition, records in messages.items():
                for record in records:
                    prediction_data = record.value
                    st.json(prediction_data) # Display raw JSON for debugging
                    
                    # Add to list (or update based on ID)
                    predictions_list.append(prediction_data)

                    # Keep only the last N predictions to avoid memory issues
                    if len(predictions_list) > 50: # Adjust this number as needed
                        predictions_list.pop(0) # Remove the oldest
            
            # Update the DataFrame and display
            if predictions_list:
                df_predictions = pd.DataFrame(predictions_list)
                # You might want to select/reorder columns for display
                # df_predictions = df_predictions[['id', 'feature1', 'predicted_label', 'timestamp']]
                data_container.dataframe(df_predictions, use_container_width=True)
            else:
                data_container.write("Waiting for predictions...")
        else:
            data_container.write("Waiting for predictions...")
            time.sleep(1) # Sleep to prevent busy-waiting if no messages

else:
    st.warning("Dashboard not connected to Kafka. Check logs for details.")