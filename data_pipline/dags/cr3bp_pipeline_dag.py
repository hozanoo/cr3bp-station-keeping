"""
Airflow DAG for the CR3BP HNN data pipeline.

This DAG orchestrates the daily generation and loading of CR3BP
simulation data:

1. Run a batch of CR3BP simulations in 3D using the DOP853 integrator.
2. Store raw trajectories in ``data/raw``.
3. Load the resulting files into PostgreSQL.
4. Refresh the HNN training view.

The code below is a skeleton and may need adjustments to match the
actual Airflow version and deployment setup.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from data_pipeline.extract.cr3bp_batch_exporter import (
    CR3BPSimulationConfig,
    generate_cr3bp_batch,
)
from data_pipeline.load.loader_postgres import (
    load_simulation_batch,
)
# Optionally: a small helper to refresh the materialized view


DEFAULT_ARGS = {
    "owner": "cr3bp_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _run_cr3bp_batch(**context):
    """
    Airflow callable to generate a batch of CR3BP simulations.

    This function should construct a :class:`CR3BPSimulationConfig`
    and pass it to :func:`generate_cr3bp_batch`. The list of generated
    file paths can be pushed to XCom for the loader task.
    """
    # TODO: Read parameters from Airflow Variables or DAG config
    config = CR3BPSimulationConfig(
        scenario_name="earth-moon-L1-3D",
        n_simulations=20,
        n_steps=4000,
        dt=0.01,
        output_dir=Path("/opt/airflow/data/raw"),
    )
    file_paths = generate_cr3bp_batch(config)
    context["ti"].xcom_push(key="cr3bp_raw_files", value=[str(p) for p in file_paths])


def _load_batch_into_db(**context):
    """
    Airflow callable to load generated simulation files into PostgreSQL.
    """
    raw_files = context["ti"].xcom_pull(key="cr3bp_raw_files", task_ids="generate_cr3bp_batch")
    paths = [Path(p) for p in raw_files]
    load_simulation_batch(paths, scenario_name="earth-moon-L1-3D")


with DAG(
    dag_id="cr3bp_hnn_data_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily CR3BP simulation generation and loading for HNN training.",
    schedule_interval="0 2 * * *",  # every day at 02:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cr3bp", "hnn", "simulation"],
) as dag:

    generate_task = PythonOperator(
        task_id="generate_cr3bp_batch",
        python_callable=_run_cr3bp_batch,
        provide_context=True,
    )

    load_task = PythonOperator(
        task_id="load_cr3bp_batch",
        python_callable=_load_batch_into_db,
        provide_context=True,
    )

    generate_task >> load_task
