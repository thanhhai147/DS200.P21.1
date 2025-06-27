import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import time
from collections import deque

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Real-time Dashboard", page_icon="📊", layout="wide")
st.title("📊 Real-time Spark Model Predictions & Metrics")
st.write("Monitoring predictions and performance from the Spark inference pipeline.")

# --- Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = 'kafka:29092'
PREDICTED_TOPIC = 'predicted_test_data'
METRICS_TOPIC = 'realtime_metrics'

# --- Metrics Global Storage ---
latest_metrics = None
metrics_history = deque(maxlen=10)
latest_metrics_timestamp = None

# --- Kafka Consumer Initialization ---
@st.cache_resource
def get_kafka_consumer(topic_name, group_id):
    return KafkaConsumer(
        topic_name,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=1000
    )

predictions_consumer = get_kafka_consumer(PREDICTED_TOPIC, 'streamlit-predictions-group')
metrics_consumer = get_kafka_consumer(METRICS_TOPIC, 'streamlit-metrics-group')

# --- UI Columns ---
col1, col2 = st.columns(2)
with col1:
    st.header("Latest Predictions")
    predictions_container = st.empty()
    predictions_list = deque(maxlen=50)

with col2:
    st.header("Real-time Model Metrics")
    accuracy_per_label_placeholder = st.empty()
    exact_match_accuracy_placeholder = st.empty()
    f1_micro_placeholder = st.empty()
    f1_macro_placeholder = st.empty()
    metrics_history_placeholder = st.empty()

# --- Main Loop for Streamlit UI ---
while True:
    messages_preds = predictions_consumer.poll(timeout_ms=100, max_records=10)
    for _, records in messages_preds.items():
        for record in records:
            predictions_list.append(record.value)
    
    messages_metrics = metrics_consumer.poll(timeout_ms=200, max_records=1)
    for _, records in messages_metrics.items():
        for record in records:
            data = record.value
            if data.get("timestamp") != latest_metrics_timestamp:
                latest_metrics_timestamp = data["timestamp"]
                latest_metrics = data
                metrics_history.append(data)

    if predictions_list:
        df_predictions = pd.DataFrame(list(predictions_list))
        display_cols = ['processing_timestamp', 'comment', 'n_star', 'label', 'predicted_label']
        df_predictions = df_predictions[display_cols]
        predictions_container.dataframe(df_predictions, use_container_width=True, hide_index=True)
    else:
        predictions_container.write("Waiting for predictions...")

    if latest_metrics:
        accuracy_per_label_placeholder.metric("Accuracy per label", f"{latest_metrics.get('accuracy_per_label', 0.0):.4f}")
        exact_match_accuracy_placeholder.metric("Accuracy exact match", f"{latest_metrics.get('exact_match_accuracy', 0.0):.4f}")
        f1_micro_placeholder.metric("F1-micro", f"{latest_metrics.get('f1_micro', 0.0):.4f}")
        f1_macro_placeholder.metric("F1-macro", f"{latest_metrics.get('f1_macro', 0.0):.4f}")

        df_metrics_history = pd.DataFrame(list(metrics_history))
        if not df_metrics_history.empty:
            df_metrics_history_display = df_metrics_history[["timestamp", "batch_id", "total_records", "accuracy_per_label", "exact_match_accuracy", "f1_micro", "f1_macro"]]
            metrics_history_placeholder.dataframe(df_metrics_history_display, use_container_width=True, hide_index=True)
    else:
        metrics_history_placeholder.write("Waiting for metrics...")

    time.sleep(2)
