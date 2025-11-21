"""
Airflow DAG definitions for the CR3BP data pipeline.

All DAG modules in this package should be importable by the Airflow
scheduler. Keep DAG definitions small and delegate heavy work to the
`data_pipeline.extract` and `data_pipeline.load` modules.
"""
