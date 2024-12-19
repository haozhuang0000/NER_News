from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from ingestion.sources.latest_news import update_latest_news_database

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
        task_id="retrieve and write", python_callable=update_latest_news_database
    )
