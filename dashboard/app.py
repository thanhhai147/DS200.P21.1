# dashboard/dashboard.py (UPDATED for Real-time Metrics Display)
import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import time
from collections import deque # For storing a limited history of metrics

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Spark Model Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Real-time Spark Model Predictions & Metrics")
st.write("Monitoring predictions and performance from the Spark inference pipeline.")

# --- Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = 'kafka:29092'
PREDICTED_TOPIC = 'predicted_test_data' # Predictions from Spark
METRICS_TOPIC = 'realtime_metrics'     # Metrics from Spark

# --- Kafka Consumer Initialization ---
@st.cache_resource
def get_kafka_consumer(topic_name, group_id):
    """
    Initializes and returns a KafkaConsumer instance for a given topic.
    @st.cache_resource ensures the consumer is created only once.
    """
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='latest', # Start consuming from the latest offset
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        st.success(f"Successfully connected to Kafka topic: {topic_name}")
        return consumer
    except Exception as e:
        st.error(f"Error connecting to Kafka topic '{topic_name}': {e}")
        st.warning(f"Please ensure Kafka is running and topic '{topic_name}' exists.")
        return None

# Get consumers for both predictions and metrics
predictions_consumer = get_kafka_consumer(PREDICTED_TOPIC, 'streamlit-predictions-group')
metrics_consumer = get_kafka_consumer(METRICS_TOPIC, 'streamlit-metrics-group')


# --- Display Data ---
if predictions_consumer:
    # Use columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Latest Predictions:")
        # Placeholder for the data table
        predictions_container = st.empty()
        
        # Initialize an empty list to store predictions
        predictions_list = deque(maxlen=50) # Use deque for efficient appending/popping

    with col2:
        st.header("Real-time Model Metrics:")
        accuracy_placeholder = st.empty()
        f1_score_placeholder = st.empty()
        metrics_history_placeholder = st.empty() # For a small history table of metrics

        # Initialize metrics display
        latest_accuracy = 0.0
        latest_f1_score = 0.0
        metrics_history_list = deque(maxlen=10) # Keep last 10 batches of metrics


    # Simple loop to continuously read from Kafka
    while True:
        # --- Process Predictions ---
        messages_preds = predictions_consumer.poll(timeout_ms=100)
        if messages_preds:
            for topic_partition, records in messages_preds.items():
                for record in records:
                    prediction_data = record.value
                    # st.json(prediction_data) # Display raw JSON for debugging

                    predictions_list.append(prediction_data)
            
            # Update the DataFrame and display
            if predictions_list:
                df_predictions = pd.DataFrame(list(predictions_list))
                # Select and reorder columns for display
                display_cols = ['processing_timestamp', 'comment', 'n_star', 'label', 'predicted_sentiment_string', 'positive_probability']
                df_predictions = df_predictions[display_cols]
                predictions_container.dataframe(df_predictions, use_container_width=True, hide_index=True)
            else:
                predictions_container.write("Waiting for predictions...")
        else:
            if not predictions_list: # Only show waiting message if no data yet
                predictions_container.write("Waiting for predictions...")
        
        # --- Process Metrics ---
        if metrics_consumer:
            messages_metrics = metrics_consumer.poll(timeout_ms=100)
            if messages_metrics:
                for topic_partition, records in messages_metrics.items():
                    for record in records:
                        metrics_data = record.value
                        latest_accuracy = metrics_data.get('accuracy', 0.0)
                        latest_f1_score = metrics_data.get('f1_score', 0.0)
                        metrics_history_list.append(metrics_data) # Add to history
            
            # Update metrics display
            accuracy_placeholder.metric("Accuracy", f"{latest_accuracy:.4f}")
            f1_score_placeholder.metric("F1-Score", f"{latest_f1_score:.4f}")
            
            if metrics_history_list:
                df_metrics_history = pd.DataFrame(list(metrics_history_list))
                df_metrics_history_display = df_metrics_history[['timestamp', 'batch_id', 'total_records', 'accuracy', 'f1_score']]
                metrics_history_placeholder.dataframe(df_metrics_history_display, use_container_width=True, hide_index=True)
            else:
                metrics_history_placeholder.write("Waiting for metrics...")
        else:
            metrics_history_placeholder.write("Metrics dashboard not connected to Kafka.")

        time.sleep(1) # Sleep to prevent busy-waiting

else:
    st.warning("Dashboard not connected to Kafka. Check logs for details.")