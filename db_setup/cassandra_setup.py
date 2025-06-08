# db_setup/cassandra_setup.py
from cassandra.cluster import Cluster, ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CASSANDRA_HOST = ['cassandra']
CASSANDRA_PORT = 9042
CASSANDRA_USER = None
CASSANDRA_PASSWORD = None

KEYSPACE_NAME = "my_project_keyspace"
TRAIN_DATA_TABLE = "raw_train_data"
TEST_DATA_TABLE = "raw_test_data"

def create_cassandra_schema():
    cluster = None
    session = None
    max_retries = 10
    retry_delay_sec = 10

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT}...")
            
            # For default Cassandra image, no auth needed. If you added user/pass, configure here:
            # auth_provider = PlainTextAuthProvider(username=CASSANDRA_USER, password=CASSANDRA_PASSWORD)
            cluster = Cluster(CASSANDRA_HOST, port=CASSANDRA_PORT) #, auth_provider=auth_provider)
            session = cluster.connect()
            logger.info("Successfully connected to Cassandra.")

            # Create Keyspace
            keyspace_query = f"""
            CREATE KEYSPACE IF NOT EXISTS {KEYSPACE_NAME}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
            """
            session.execute(keyspace_query)
            logger.info(f"Keyspace '{KEYSPACE_NAME}' created or already exists.")
            session.set_keyspace(KEYSPACE_NAME)

            # Create raw_train_data table
            train_table_query = f"""
            CREATE TABLE IF NOT EXISTS {TRAIN_DATA_TABLE} (
                id text PRIMARY KEY,        -- Generated unique ID for each record
                comment text,
                n_star int,
                date_time timestamp,        -- Store as timestamp after parsing
                label text,
                processing_timestamp timestamp -- Timestamp when Spark processed it
            );
            """
            session.execute(train_table_query)
            logger.info(f"Table '{TRAIN_DATA_TABLE}' created or already exists.")

            # Create raw_test_data table
            test_table_query = f"""
            CREATE TABLE IF NOT EXISTS {TEST_DATA_TABLE} (
                id text PRIMARY KEY,        -- Generated unique ID for each record
                comment text,
                n_star int,
                date_time timestamp,
                label text,                 -- Test data might or might not have a ground truth label
                processing_timestamp timestamp
            );
            """
            session.execute(test_table_query)
            logger.info(f"Table '{TEST_DATA_TABLE}' created or already exists.")

            logger.info("Cassandra schema setup complete!")
            return # Success, exit function
        except Exception as e:
            logger.warning(f"Cassandra connection or schema setup failed: {e}. Retrying in {retry_delay_sec}s...")
            time.sleep(retry_delay_sec)
        finally:
            if session:
                session.shutdown()
            if cluster:
                cluster.shutdown()
    
    logger.error(f"Failed to setup Cassandra schema after {max_retries} attempts.")
    raise Exception("Cassandra schema setup failed.")

if __name__ == "__main__":
    create_cassandra_schema()