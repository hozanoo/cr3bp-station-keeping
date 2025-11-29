# data_pipeline/load/loader_postgres.py
"""
Loader for CR3BP trajectory CSVs into PostgreSQL.

Creates tables, manages foreign keys, and loads batch CSV trajectories.

Schema (in schema "cr3bp"):
    cr3bp_system
    cr3bp_lagrange_point
    cr3bp_simulation_run
    cr3bp_trajectory_sample
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from data_pipeline.load.db_connection import db_cursor
from sim_rl.cr3bp.scenarios import SCENARIOS

# Root where raw CSV batches land (mounted by Docker)
RAW_EXPORT_ROOT = (Path(__file__).resolve().parent.parent / "raw_exports").resolve()


# ============================================================
# 1) SCHEMA CREATION
# ============================================================
def ensure_schema() -> None:
    """Create tables if they do not exist."""

    ddl_system = """
    CREATE TABLE IF NOT EXISTS cr3bp_system (
        system_id     SERIAL PRIMARY KEY,
        name          TEXT UNIQUE NOT NULL,
        frame_default TEXT NOT NULL CHECK (
            frame_default IN ('rotating', 'inertial', 'barycentric')
        )
    );
    """

    ddl_lpoint = """
    CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
        lagrange_point_id SERIAL PRIMARY KEY,
        system_id         INT NOT NULL REFERENCES cr3bp_system(system_id),
        name              TEXT NOT NULL,
        UNIQUE (system_id, name)
    );
    """

    ddl_run = """
    CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
        run_id                 UUID PRIMARY KEY,
        system_id              INT NOT NULL REFERENCES cr3bp_system(system_id),
        lagrange_point_id      INT NOT NULL REFERENCES cr3bp_lagrange_point(lagrange_point_id),
        scenario_name          TEXT NOT NULL,
        created_at             TIMESTAMPTZ NOT NULL,
        source_file            TEXT NOT NULL,

        integrator             TEXT,
        step_size              DOUBLE PRECISION,
        terminated             BOOLEAN NOT NULL DEFAULT FALSE,
        termination_reason     TEXT,
        initial_condition_type TEXT
    );
    """

    ddl_sample = """
    CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
        run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(run_id) ON DELETE CASCADE,
        step   INT NOT NULL,
        t      DOUBLE PRECISION NOT NULL,

        x      DOUBLE PRECISION NOT NULL,
        y      DOUBLE PRECISION NOT NULL,
        z      DOUBLE PRECISION NOT NULL,

        vx     DOUBLE PRECISION NOT NULL,
        vy     DOUBLE PRECISION NOT NULL,
        vz     DOUBLE PRECISION NOT NULL,

        ax     DOUBLE PRECISION NOT NULL,
        ay     DOUBLE PRECISION NOT NULL,
        az     DOUBLE PRECISION NOT NULL,

        PRIMARY KEY (run_id, step)
    );
    """

    with db_cursor(autocommit=True) as cur:
        cur.execute(ddl_system)
        cur.execute(ddl_lpoint)
        cur.execute(ddl_run)
        cur.execute(ddl_sample)


# ============================================================
# 2) HELPERS
# ============================================================
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
        """
        SELECT lagrange_point_id
        FROM cr3bp_lagrange_point
        WHERE system_id = %s AND name = %s;
        """,
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


def parse_metadata_from_filename(csv_path: Path) -> Tuple[str, bool, str]:
    """
    Parse initial_condition_type and termination_reason from filenames like:

        traj_000_l1_cloud_max_steps.csv
        traj_001_halo_seed_escape.csv

    Returns
    -------
    ic_type : str
    terminated : bool
        True iff termination_reason != "max_steps".
    termination_reason : str
    """
    stem = csv_path.stem  # e.g. "traj_000_l1_cloud_max_steps"
    parts = stem.split("_")

    if len(parts) < 4 or parts[0] != "traj":
        raise ValueError(f"Unexpected trajectory filename format: {csv_path.name!r}")

    termination_reason = parts[-1]
    ic_type = "_".join(parts[2:-1])
    terminated = termination_reason != "max_steps"
    return ic_type, terminated, termination_reason


# ============================================================
# 3) INSERT CSV → DB
# ============================================================
def insert_run_and_trajectory(csv_path: Path, scenario_name: str) -> uuid.UUID:
    """
    Insert a single CSV trajectory file into cr3bp_* tables.

    Parameters
    ----------
    csv_path:
        File to load.
    scenario_name:
        Name matching SCENARIOS dict.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined in SCENARIOS.")

    scenario = SCENARIOS[scenario_name]
    df = pd.read_csv(csv_path)

    expected_cols = {"t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path} missing required columns: {sorted(expected_cols)}"
        )

    if len(df) < 2:
        raise ValueError(f"CSV {csv_path} is too short for a useful trajectory.")

    step_size = float(df["t"].iloc[1] - df["t"].iloc[0])
    integrator = "DOP853"

    ic_type, terminated, termination_reason = parse_metadata_from_filename(csv_path)

    run_id = uuid.uuid4()
    created_at = datetime.utcnow()
    source_file = str(csv_path)

    with db_cursor(autocommit=True) as cur:
        ensure_schema()

        system_id = get_or_create_system(cur, scenario.system, frame_default="rotating")
        lagrange_point_id = get_or_create_lagrange_point(
            cur, system_id, scenario.lagrange_point
        )

        cur.execute(
            """
            INSERT INTO cr3bp_simulation_run (
                run_id,
                system_id,
                lagrange_point_id,
                scenario_name,
                created_at,
                source_file,
                integrator,
                step_size,
                terminated,
                termination_reason,
                initial_condition_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                run_id,
                system_id,
                lagrange_point_id,
                scenario_name,
                created_at,
                source_file,
                integrator,
                step_size,
                terminated,
                termination_reason,
                ic_type,
            ),
        )

        for step_idx, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO cr3bp_trajectory_sample (
                    run_id,
                    step,
                    t,
                    x, y, z,
                    vx, vy, vz,
                    ax, ay, az
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    run_id,
                    int(step_idx),
                    float(row["t"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row["vx"]),
                    float(row["vy"]),
                    float(row["vz"]),
                    float(row["ax"]),
                    float(row["ay"]),
                    float(row["az"]),
                ),
            )

    return run_id


# ============================================================
# 4) FILE DISCOVERY
# ============================================================
def find_csv_files_for_date(
    scenario_name: str, date_str: str | None = None
) -> List[Path]:
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y%m%d")

    base_dir = RAW_EXPORT_ROOT / scenario_name / date_str
    if not base_dir.exists():
        return []

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    search_dir = sorted(subdirs)[-1] if subdirs else base_dir

    return sorted(search_dir.glob("traj_*.csv"))


# ============================================================
# 5) HIGH-LEVEL ENTRY POINTS
# ============================================================
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
