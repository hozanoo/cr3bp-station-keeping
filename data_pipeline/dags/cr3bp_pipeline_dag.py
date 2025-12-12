# data_pipeline/dags/cr3bp_pipeline_dag.py
"""
Airflow DAG for the CR3BP data pipeline.

This DAG orchestrates two main steps:

1. Generate raw CR3BP trajectory batches as CSV files using the
   batch exporter. This step produces both standard L1 / halo-seed
   runs and halo-reference runs along the precomputed halo orbit.

2. Load all generated CSV files for the current UTC date into
   PostgreSQL via the loader module, populating the normalized
   CR3BP schema inside the ``cr3bp`` schema.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Ensure /opt/airflow (project root) is in sys.path for data_pipeline imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> /opt/airflow
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from data_pipeline.extract.cr3bp_batch_exporter import generate_batch_simulations
from data_pipeline.load.loader_postgres import load_raw_export_directory


def task_generate_simulations(**context) -> None:
    """
    Airflow task wrapper for the CR3BP batch exporter.

    Generates both standard L1 / halo-seed trajectories and
    halo-reference trajectories for the current run. All CSV
    files are written below ``data_pipeline/raw_exports``.
    """
    paths = generate_batch_simulations()
    print(f"[CR3BP] Generated {len(paths)} trajectories in total.")
    if paths:
        print(f"[CR3BP] Export directory: {paths[0].parent}")


def task_load_csvs(**context) -> None:
    """
    Airflow task wrapper for the CR3BP PostgreSQL loader.

    Loads all CSV trajectories for the configured scenarios on the
    current UTC date into the normalized CR3BP schema.
    """
    run_ids = load_raw_export_directory()
    print(f"[CR3BP] Inserted {len(run_ids)} simulation runs into PostgreSQL.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
default_args = {
    "owner": "cr3bp",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cr3bp_hourly_pipeline",
    description="Generate CR3BP trajectories and load them into PostgreSQL.",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["cr3bp", "hnn", "station-keeping"],
) as dag:

    generate = PythonOperator(
        task_id="generate_simulations",
        python_callable=task_generate_simulations,
    )

    load_csv = PythonOperator(
        task_id="load_csvs_into_postgres",
        python_callable=task_load_csvs,
    )

    generate >> load_csv
