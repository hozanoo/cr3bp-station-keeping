# data_pipline/dags/cr3bp_pipeline_dag.py

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from data_pipeline.extract.cr3bp_batch_exporter import generate_batch_simulations
from data_pipeline.load.loader_postgres import load_raw_export_directory

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cr3bp_daily_pipeline",
    description="Daily CR3BP batch simulations → raw CSV → Postgres load",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 3 * * *",   # daily at 03:00
    catchup=False,
    max_active_runs=1,
) as dag:

    def task_generate_simulations():
        # This function is implemented in extract/cr3bp_batch_exporter.py
        generate_batch_simulations()

    def task_load_csvs():
        # Loads ALL CSVs found in data_pipline/data/raw_exports/
        load_raw_export_directory()

    generate = PythonOperator(
        task_id="generate_simulations",
        python_callable=task_generate_simulations,
    )

    load_csv = PythonOperator(
        task_id="load_csvs_into_postgres",
        python_callable=task_load_csvs,
    )

    generate >> load_csv
