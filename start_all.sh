# start_all.sh

#!/bin/bash

echo "--- Building and Starting Docker Compose Services ---"
# Stops and removes old containers/volumes if they exist, then builds and starts
docker-compose down --rmi all -v --remove-orphans # Clean shutdown
docker system prune -a
docker-compose up -d --build

echo "--- Waiting for core services (Kafka, Cassandra, Spark Master) to be healthy ---"
for i in {1..3}; do 
  KAFKA_HEALTH=$(docker-compose inspect --format '{{.State.Health.Status}}' kafka 2>/dev/null)
  CASSANDRA_HEALTH=$(docker-compose inspect --format '{{.State.Health.Status}}' cassandra 2>/dev/null)
  echo "$KAFKA_HEALTH"
  echo "$CASSANDRA_HEALTH"
  # Check if all critical services are healthy/running
  if [[ "$KAFKA_HEALTH" == "healthy" && "$CASSANDRA_HEALTH" == "healthy" ]]; then
    echo "Kafka and Cassandra are healthy!"
    break
  fi
  echo "Waiting for services to be healthy... ($i/3)"
  sleep 5
done

echo "--- Creating Kafka Topics ---"
DOCKER_KAFKA_COMMAND="docker-compose exec kafka kafka-topics"

docker-compose exec kafka kafka-topics --create --topic raw_train_data --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1 --if-not-exists
docker-compose exec kafka kafka-topics --create --topic raw_test_data --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1 --if-not-exists
docker-compose exec kafka kafka-topics --create --topic predicted_test_data --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1 --if-not-exists
docker-compose exec kafka kafka-topics --create --topic realtime_metrics --bootstrap-server kafka:29092 --partitions 1 --replication-factor 1 --if-not-exists

echo "--- Listing Kafka Topics to Verify ---"
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo "--- Setting Permissions for Spark Jobs Directory ---"
# Ensure the host mounted volume has write permissions for Spark jobs
sudo chmod -R 777 spark_jobs

echo "--- Run spark workers ---"

# # 1. Submit process_train_data.py
# echo "Submitting process_train_data.py (Kafka to Cassandra ingestion for training data)..."
# docker-compose exec -T spark-master /opt/bitnami/spark/bin/spark-submit \
#     --master spark://spark-master:7077 \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
#     --driver-memory 2g \
#     --executor-memory 2g \
#     --total-executor-cores 1 \
#     /opt/bitnami/spark/jobs/process_train_data.py &
# PROCESS_TRAIN_PID=$!
# echo "process_train_data.py submitted with PID: $PROCESS_TRAIN_PID"

# # 2. Submit continuous_model_trainer.py
# echo "Submitting continuous_model_trainer.py (Periodic model training from Cassandra)..."
# docker-compose exec -T spark-master /opt/bitnami/spark/bin/spark-submit \
#     --master spark://spark-master:7077 \
#     --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
#     --driver-memory 4g \
#     --executor-memory 4g \
#     --total-executor-cores 1 \
#     /opt/bitnami/spark/jobs/continuous_model_trainer.py &
# PROCESS_TRAINER_PID=$!
# echo "continuous_model_trainer.py submitted with PID: $PROCESS_TRAINER_PID"

# # 3. Submit process_test_data.py
# echo "Submitting process_test_data.py (Kafka to prediction/Cassandra ingestion for test data)..."
# docker-compose exec -T spark-master /opt/bitnami/spark/bin/spark-submit \
#     --master spark://spark-master:7077 \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
#     --driver-memory 4g \
#     --executor-memory 4g \
#     --total-executor-cores 1 \
#     /opt/bitnami/spark/jobs/process_test_data.py &
# PROCESS_TEST_PID=$!
# echo "process_test_data.py submitted with PID: $PROCESS_TEST_PID"

echo "--- Setup completed ---"