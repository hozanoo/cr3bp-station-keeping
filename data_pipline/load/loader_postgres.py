"""
Loading CR3BP simulation data into PostgreSQL.

This module implements the "L" of the ELT pipeline. It takes raw
simulation files (e.g. CSV or Parquet) produced by the extract layer
and inserts them into a relational schema suited for HNN training
and analysis.

Recommended schema (can be adapted as needed):

- sim.simulation_run
- sim.simulation_state

The functions here are skeletons; they document the intended behaviour
but do not yet implement the full SQL/ORM logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from .db_connection import create_db_engine


def ensure_simulation_tables(engine: Engine) -> None:
    """
    Ensure that the base tables for simulation data exist.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the target PostgreSQL database.

    Notes
    -----
    There are two main implementation options:

    - Use SQLAlchemy ORM / Table metadata to create tables via
      ``metadata.create_all(engine)``.
    - Use pure SQL DDL statements (``CREATE TABLE IF NOT EXISTS``).

    The actual table definitions should match the structure you plan
    to use in your HNN training views.
    """
    # TODO: Implement table creation logic (SQLAlchemy or raw SQL).
    raise NotImplementedError("Table creation for simulation schema is not implemented yet.")


def load_simulation_file(
    file_path: Path,
    engine: Engine,
    scenario_name: str,
    sim_batch_id: Optional[str] = None,
) -> None:
    """
    Load a single CR3BP simulation file into the database.

    Parameters
    ----------
    file_path:
        Path to the raw simulation file (CSV or Parquet).
    engine:
        SQLAlchemy engine connected to PostgreSQL.
    scenario_name:
        Name of the scenario this simulation belongs to, e.g.
        ``"earth-moon-L1-3D"``.
    sim_batch_id:
        Optional batch identifier to link multiple runs that were
        generated together in one Airflow task.

    Notes
    -----
    The function is expected to:

    - read the file into a :class:`pandas.DataFrame`,
    - normalise column names as needed,
    - insert a row into ``sim.simulation_run``,
    - insert all time steps into ``sim.simulation_state``.
    """
    # TODO: Implement file loading and insertion logic.
    # Example: df = pd.read_parquet(file_path) or read_csv(...)
    raise NotImplementedError("Single-file load is not implemented yet.")


def load_simulation_batch(
    files: Iterable[Path],
    engine: Optional[Engine] = None,
    scenario_name: str = "earth-moon-L1-3D",
    sim_batch_id: Optional[str] = None,
) -> None:
    """
    Load a batch of simulation files into PostgreSQL.

    Parameters
    ----------
    files:
        Iterable of file paths produced by the extract step.
    engine:
        Optional existing engine. If not provided, a new one is created
        via :func:`data_pipeline.load.db_connection.create_db_engine`.
    scenario_name:
        Scenario label shared by the batch (can be overridden per-file
        if needed).
    sim_batch_id:
        Optional identifier for the whole batch, useful for later
        filtering or retraining.

    Notes
    -----
    This function is a thin orchestrator that iterates over all files
    and calls :func:`load_simulation_file` for each of them.
    """
    if engine is None:
        engine = create_db_engine()

    # TODO: Implement batch loop and call load_simulation_file for each path.
    raise NotImplementedError("Batch loading is not implemented yet.")
