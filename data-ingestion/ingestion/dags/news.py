from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from ingestion.main import manage_data_processing

defaults = {
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=60),
}


with DAG(
    dag_id="data_ingestion",
    default_args=defaults,
    schedule_interval="@daily",
    catchup=False,
    description="Ingests data from external sources and writes to Mongo",
):

    retrieve_news = PythonOperator(
        task_id="start_ingestion", python_callable=manage_data_processing
    )
