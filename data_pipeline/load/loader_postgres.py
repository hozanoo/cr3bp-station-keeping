# data_pipeline/load/loader_postgres.py
"""
Loader for CR3BP trajectory CSVs into PostgreSQL.
Creates tables, foreign keys, and loads all trajectory CSVs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from data_pipeline.load.db_connection import db_cursor
from sim_rl.cr3bp.scenarios import SCENARIOS


RAW_EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"


# -----------------------------------------------------------
# 1) SCHEMA FIXED — using system_id / lagrange_point_id / run_id
# -----------------------------------------------------------
def ensure_schema() -> None:
    """Create the required tables if they do not exist."""

    ddl_system = """
    CREATE TABLE IF NOT EXISTS cr3bp_system (
        system_id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        frame_default TEXT NOT NULL CHECK (
            frame_default IN ('rotating', 'inertial', 'barycentric')
        )
    );
    """

    ddl_lpoint = """
    CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
        lagrange_point_id SERIAL PRIMARY KEY,
        system_id INT NOT NULL REFERENCES cr3bp_system(system_id),
        name TEXT NOT NULL,
        UNIQUE(system_id, name)
    );
    """

    ddl_run = """
    CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
        run_id UUID PRIMARY KEY,
        system_id INT NOT NULL REFERENCES cr3bp_system(system_id),
        lagrange_point_id INT NOT NULL REFERENCES cr3bp_lagrange_point(lagrange_point_id),
        scenario_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        source_file TEXT NOT NULL
    );
    """

    ddl_sample = """
    CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
        run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(run_id),
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


# -----------------------------------------------------------
# HELPERS TO GET OR CREATE KEYS
# -----------------------------------------------------------
def get_or_create_system(cur, name: str, frame_default: str = "rotating") -> int:
    cur.execute("SELECT system_id FROM cr3bp_system WHERE name = %s;", (name,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO cr3bp_system (name, frame_default)
        VALUES (%s, %s)
        RETURNING system_id;
        """,
        (name, frame_default),
    )
    return cur.fetchone()[0]


def get_or_create_lagrange_point(cur, system_id: int, name: str) -> int:
    cur.execute(
        "SELECT lagrange_point_id FROM cr3bp_lagrange_point WHERE system_id = %s AND name = %s;",
        (system_id, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO cr3bp_lagrange_point (system_id, name)
        VALUES (%s, %s)
        RETURNING lagrange_point_id;
        """,
        (system_id, name),
    )
    return cur.fetchone()[0]


# -----------------------------------------------------------
# CSV LOADING
# -----------------------------------------------------------
def insert_run_and_trajectory(csv_path: Path, scenario_name: str) -> uuid.UUID:
    """Insert a single trajectory CSV into the database."""
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined in SCENARIOS.")

    scenario = SCENARIOS[scenario_name]
    df = pd.read_csv(csv_path)

    expected_cols = {"t", "x", "y", "z", "vx", "vy", "vz"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} missing required columns: {expected_cols}")

    run_id = uuid.uuid4()
    created_at = datetime.utcnow()

    with db_cursor(autocommit=False) as cur:
        system_id = get_or_create_system(cur, scenario.system)
        lpoint_id = get_or_create_lagrange_point(cur, system_id, scenario.lagrange_point)

        cur.execute(
            """
            INSERT INTO cr3bp_simulation_run (
                run_id, system_id, lagrange_point_id,
                scenario_name, created_at, source_file
            ) VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (str(run_id), system_id, lpoint_id, scenario.name, created_at, str(csv_path)),
        )

        records = [
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
            for i, row in df.iterrows()
        ]

        cur.executemany(
            """
            INSERT INTO cr3bp_trajectory_sample (
                run_id, step, t,
                x, y, z,
                vx, vy, vz
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            records,
        )

        cur.connection.commit()

    return run_id


# -----------------------------------------------------------
# FIND FILES
# -----------------------------------------------------------
def find_csv_files_for_date(scenario_name: str, date_str: str | None = None) -> List[Path]:
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y%m%d")

    base_dir = RAW_EXPORT_ROOT / scenario_name / date_str
    if not base_dir.exists():
        return []

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    search_dir = sorted(subdirs)[-1] if subdirs else base_dir

    return sorted(search_dir.glob("traj_*.csv"))


# -----------------------------------------------------------
# ENTRY POINTS
# -----------------------------------------------------------
def load_batch_for_date(scenario_name: str, date_str: str | None = None) -> List[uuid.UUID]:
    ensure_schema()

    csv_paths = find_csv_files_for_date(scenario_name, date_str)
    if not csv_paths:
        print(f"[INFO] No CSVs for scenario={scenario_name}, date={date_str}")
        return []

    return [insert_run_and_trajectory(path, scenario_name) for path in csv_paths]


def load_raw_export_directory() -> List[uuid.UUID]:
    today = datetime.utcnow().strftime("%Y%m%d")
    return load_batch_for_date("earth-moon-L1-3D", today)


if __name__ == "__main__":
    ids = load_raw_export_directory()
    print(f"Inserted {len(ids)} simulation runs.")
