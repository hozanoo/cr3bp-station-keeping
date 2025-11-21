# data_pipline/load/loader_postgres.py
"""
Loader for CR3BP trajectory CSVs into the HNN training database.

This module reads raw trajectory files (CSV) produced by
``data_pipline.extract.cr3bp_batch_exporter`` and loads them into a
normalized PostgreSQL schema.

Schema overview
---------------

The loader expects (or creates) the following tables:

- cr3bp_system
    - id SERIAL PRIMARY KEY
    - name TEXT UNIQUE NOT NULL
    - frame_default TEXT NOT NULL
- cr3bp_lagrange_point
    - id SERIAL PRIMARY KEY
    - system_id INT NOT NULL REFERENCES cr3bp_system(id)
    - name TEXT NOT NULL
    - UNIQUE (system_id, name)
- cr3bp_simulation_run
    - id UUID PRIMARY KEY
    - system_id INT NOT NULL REFERENCES cr3bp_system(id)
    - lagrange_point_id INT NOT NULL REFERENCES cr3bp_lagrange_point(id)
    - scenario_name TEXT NOT NULL
    - created_at TIMESTAMPTZ NOT NULL
    - source_file TEXT NOT NULL
- cr3bp_trajectory_sample
    - run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(id)
    - step INT NOT NULL
    - t DOUBLE PRECISION NOT NULL
    - x DOUBLE PRECISION NOT NULL
    - y DOUBLE PRECISION NOT NULL
    - z DOUBLE PRECISION NOT NULL
    - vx DOUBLE PRECISION NOT NULL
    - vy DOUBLE PRECISION NOT NULL
    - vz DOUBLE PRECISION NOT NULL
    - PRIMARY KEY (run_id, step)

The loader can be used both from Airflow (as a PythonOperator) and
from the command line for manual imports.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from datetime import datetime
from typing import Iterable

import pandas as pd

from data_pipeline.load.db_connection import db_cursor
from sim_rl.cr3bp.scenarios import SCENARIOS


RAW_EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def ensure_schema() -> None:
    """
    Create the required tables if they do not exist.

    This function is idempotent and can be called before each load
    run without side effects.
    """
    ddl_system = """
    CREATE TABLE IF NOT EXISTS cr3bp_system (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        frame_default TEXT NOT NULL CHECK (
            frame_default IN ('rotating', 'inertial', 'barycentric')
        )
    );
    """

    ddl_lpoint = """
    CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
        id SERIAL PRIMARY KEY,
        system_id INT NOT NULL REFERENCES cr3bp_system(id),
        name TEXT NOT NULL,
        UNIQUE (system_id, name)
    );
    """

    ddl_run = """
    CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
        id UUID PRIMARY KEY,
        system_id INT NOT NULL REFERENCES cr3bp_system(id),
        lagrange_point_id INT NOT NULL REFERENCES cr3bp_lagrange_point(id),
        scenario_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        source_file TEXT NOT NULL
    );
    """

    ddl_sample = """
    CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
        run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(id),
        step INT NOT NULL,
        t DOUBLE PRECISION NOT NULL,
        x DOUBLE PRECISION NOT NULL,
        y DOUBLE PRECISION NOT NULL,
        z DOUBLE PRECISION NOT NULL,
        vx DOUBLE PRECISION NOT NULL,
        vy DOUBLE PRECISION NOT NULL,
        vz DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (run_id, step)
    );
    """

    with db_cursor(autocommit=True) as cur:
        cur.execute(ddl_system)
        cur.execute(ddl_lpoint)
        cur.execute(ddl_run)
        cur.execute(ddl_sample)


def get_or_create_system(cur, name: str, frame_default: str = "rotating") -> int:
    """
    Return the id of the system with the given name, inserting it if needed.
    """
    cur.execute(
        "SELECT id FROM cr3bp_system WHERE name = %s;",
        (name,),
    )
    row = cur.fetchone()
    if row is not None:
        return row[0]

    cur.execute(
        """
        INSERT INTO cr3bp_system (name, frame_default)
        VALUES (%s, %s)
        RETURNING id;
        """,
        (name, frame_default),
    )
    return cur.fetchone()[0]


def get_or_create_lagrange_point(cur, system_id: int, name: str) -> int:
    """
    Return the id of the Lagrange point for a given system, inserting it if needed.
    """
    cur.execute(
        """
        SELECT id
        FROM cr3bp_lagrange_point
        WHERE system_id = %s AND name = %s;
        """,
        (system_id, name),
    )
    row = cur.fetchone()
    if row is not None:
        return row[0]

    cur.execute(
        """
        INSERT INTO cr3bp_lagrange_point (system_id, name)
        VALUES (%s, %s)
        RETURNING id;
        """,
        (system_id, name),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Load logic
# ---------------------------------------------------------------------------


def insert_run_and_trajectory(csv_path: Path, scenario_name: str) -> uuid.UUID:
    """
    Insert a single trajectory CSV into the database.

    Parameters
    ----------
    csv_path:
        Path to the CSV file with columns [t, x, y, z, vx, vy, vz].
    scenario_name:
        Name of the scenario (must exist in :data:`sim_rl.cr3bp.scenarios.SCENARIOS`).

    Returns
    -------
    uuid.UUID
        The primary key of the inserted simulation run.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined in SCENARIOS.")

    scenario = SCENARIOS[scenario_name]

    df = pd.read_csv(csv_path)

    expected_cols = {"t", "x", "y", "z", "vx", "vy", "vz"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path} is missing required columns. "
            f"Expected at least {sorted(expected_cols)}, got {list(df.columns)}"
        )

    run_id = uuid.uuid4()
    created_at = datetime.utcnow()

    with db_cursor(autocommit=False) as cur:
        system_id = get_or_create_system(cur, scenario.system, frame_default="rotating")
        lpoint_id = get_or_create_lagrange_point(cur, system_id, scenario.lagrange_point)

        cur.execute(
            """
            INSERT INTO cr3bp_simulation_run (
                id, system_id, lagrange_point_id,
                scenario_name, created_at, source_file
            )
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (
                str(run_id),
                system_id,
                lpoint_id,
                scenario.name,
                created_at,
                str(csv_path),
            ),
        )

        records = []
        for i, row in df.iterrows():
            records.append(
                (
                    str(run_id),
                    int(i),
                    float(row["t"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row["vx"]),
                    float(row["vy"]),
                    float(row["vz"]),
                )
            )

        cur.executemany(
            """
            INSERT INTO cr3bp_trajectory_sample (
                run_id, step, t,
                x, y, z,
                vx, vy, vz
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            records,
        )

        cur.connection.commit()

    return run_id


def find_csv_files_for_date(
    scenario_name: str,
    date_str: str | None = None,
) -> list[Path]:
    """
    Find all trajectory CSV files for a given scenario and date.

    Parameters
    ----------
    scenario_name:
        Name of the scenario (e.g. ``"earth-moon-L1-3D"``).
    date_str:
        Date folder in ``YYYYMMDD`` format. If omitted, today's date
        is used.

    Returns
    -------
    list[pathlib.Path]
        List of CSV file paths sorted by name.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y%m%d")

    base_dir = RAW_EXPORT_ROOT / scenario_name / date_str
    if not base_dir.exists():
        return []

    csv_paths = sorted(base_dir.glob("traj_*.csv"))
    return csv_paths


def load_batch_for_date(
    scenario_name: str = "earth-moon-L1-3D",
    date_str: str | None = None,
) -> list[uuid.UUID]:
    """
    Load all trajectories for a scenario and date into the database.

    Parameters
    ----------
    scenario_name:
        Scenario used when the trajectories were generated.
    date_str:
        Date string in ``YYYYMMDD`` format. If omitted, today's date
        is used.

    Returns
    -------
    list[uuid.UUID]
        List of run ids that have been inserted.
    """
    ensure_schema()

    csv_paths = find_csv_files_for_date(scenario_name, date_str=date_str)
    if not csv_paths:
        print(
            f"[INFO] No CSV files found for scenario={scenario_name!r}, "
            f"date={date_str!r} under {RAW_EXPORT_ROOT}"
        )
        return []

    run_ids: list[uuid.UUID] = []
    for path in csv_paths:
        run_id = insert_run_and_trajectory(path, scenario_name=scenario_name)
        run_ids.append(run_id)
        print(f"[INFO] Loaded CSV {path} as run {run_id}")

    return run_ids


if __name__ == "__main__":
    scenario = "earth-moon-L1-3D"
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"Loading batch for scenario={scenario}, date={today}")
    ids = load_batch_for_date(scenario_name=scenario, date_str=today)
    print(f"Inserted {len(ids)} simulation runs.")
