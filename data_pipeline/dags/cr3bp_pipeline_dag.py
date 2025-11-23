# data_pipeline/dags/cr3bp_pipeline_dag.py

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Ensure /opt/airflow is in sys.path (for data_pipeline imports)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> /opt/airflow
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    schedule_interval="0 * * * *",  # every full hour
    catchup=False,
    max_active_runs=1,
) as dag:

    def task_generate_simulations() -> None:
        generate_batch_simulations()

    def task_load_csvs() -> None:
        load_raw_export_directory()

    # CR3BP simulation generation
    generate = PythonOperator(
        task_id="generate_simulations",
        python_callable=task_generate_simulations,
    )

    # Load CSV → Postgres
    load_csv = PythonOperator(
        task_id="load_csvs_into_postgres",
        python_callable=task_load_csvs,
    )

    generate >> load_csv
