"""
Batch exporter for CR3BP trajectories used in HNN training and analysis.

This module provides utilities to generate physics-only trajectories
from the CR3BP station-keeping environment and export them as CSV
files under a structured directory layout.

Two main families of trajectories are supported:

1. Local L1 / halo-seed cloud (physics-only, no control),
   using the environment in standard mode (``use_reference_orbit=False``).

2. Halo-reference trajectories along a precomputed halo orbit,
   using the environment in reference-orbit mode
   (``use_reference_orbit=True``) with zero control.

The exported CSV files are later ingested into PostgreSQL via the
loader module and exposed to HNN training through a dedicated view.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv


# Root directory where all raw CSV batches are stored
EXPORT_ROOT = (Path(__file__).resolve().parent.parent / "raw_exports").resolve()
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1) Local L1 / Halo-Seed Cloud Export (Standard Mode)
# ======================================================================
def simulate_single_run(
    scenario: ScenarioConfig,
    steps: int = 600,
    seed: int | None = None,
    ic_type: str = "l1_cloud",
    escape_radius: float = 0.5,
) -> Tuple[pd.DataFrame, str]:
    """
    Run a physics-only CR3BP simulation without control actions.

    The environment is configured with a chosen integrator and an
    initial-condition type. The rollout terminates on crashes,
    environment truncation, large excursions from the target region
    or non-finite states. The final crash or escape frame is not
    stored in the returned trajectory.

    Parameters
    ----------
    scenario:
        Scenario configuration used to construct the environment.
    steps:
        Maximum number of integration steps before truncation.
    seed:
        Optional seed for the environment RNG.
    ic_type:
        Initial-condition label, typically ``"l1_cloud"`` or
        ``"halo_seed"`` for three-dimensional variation.
    escape_radius:
        Threshold radius in the rotating frame beyond which the
        trajectory is classified as an escape.

    Returns
    -------
    pandas.DataFrame
        Trajectory with columns
        ``t, x, y, z, vx, vy, vz, ax, ay, az``.
    str
        Termination reason; one of:
        ``"max_steps"``, ``"escape"``, ``"crash_primary1"``,
        ``"crash_primary2"``, ``"nan"``, ``"truncated"``.
    """
    env = Cr3bpStationKeepingEnv(
        scenario=scenario,
        max_steps=steps,
        integrator="DOP853",
        seed=seed,
    )

    obs, info = env.reset(options={"ic_type": ic_type})
    dim = env.dim
    dt = env.dt

    records: list[dict[str, float]] = []

    sat = env.system.bodies[0]
    initial_row = {
        "t": 0.0,
        "x": float(sat.position[0]),
        "y": float(sat.position[1]),
        "z": float(sat.position[2]) if dim == 3 else 0.0,
        "vx": float(sat.velocity[0]),
        "vy": float(sat.velocity[1]),
        "vz": float(sat.velocity[2]) if dim == 3 else 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
    }
    records.append(initial_row)

    termination_reason = "max_steps"

    for k in range(1, steps + 1):
        action = np.zeros(dim, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]

        radius = float(np.linalg.norm(rel_pos))
        escape = radius > escape_radius

        pos = rel_pos + env.target
        vel = rel_vel
        acc = info.get("acc", np.zeros(dim, dtype=float))

        non_finite_state = (
            not np.isfinite(pos).all()
            or not np.isfinite(vel).all()
            or not np.isfinite(acc).all()
        )

        crash_p1 = bool(info.get("crash_primary1", False))
        crash_p2 = bool(info.get("crash_primary2", False))

        # Determine termination reason before appending a broken state
        if non_finite_state:
            termination_reason = "nan"
            break

        if crash_p1:
            termination_reason = "crash_primary1"
            break

        if crash_p2:
            termination_reason = "crash_primary2"
            break

        if escape:
            termination_reason = "escape"
            break

        row = {
            "t": float(k * dt),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]) if dim == 3 else 0.0,
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]) if dim == 3 else 0.0,
            "ax": float(acc[0]),
            "ay": float(acc[1]),
            "az": float(acc[2]) if dim == 3 else 0.0,
        }
        records.append(row)

        if terminated or truncated:
            termination_reason = "max_steps" if k >= steps else "truncated"
            break

    env.close()

    if len(records) < 2:
        return pd.DataFrame(), termination_reason

    df = pd.DataFrame(records)
    return df, termination_reason


def export_batch(
    scenario_name: str = "earth-moon-L1-3D",
    n_simulations: int = 3000,
    steps_per_sim: int = 600,
    halo_fraction: float = 0.5,
) -> List[Path]:
    """
    Generate multiple CR3BP trajectories in standard mode and export as CSV.

    A configurable fraction of simulations uses halo-like initial
    conditions for additional three-dimensional variation; the
    remaining share uses L1-cloud initial conditions.

    The termination reason is encoded into the filename so that the
    database loader can reconstruct metadata without re-simulating.

    Example filenames
    -----------------
    ``traj_012_l1_cloud_max_steps.csv``

    ``traj_013_halo_seed_escape.csv``

    Parameters
    ----------
    scenario_name:
        Key into the ``SCENARIOS`` registry.
    n_simulations:
        Number of independent trajectories to generate.
    steps_per_sim:
        Maximum number of integration steps per trajectory.
    halo_fraction:
        Probability for using ``"halo_seed"`` initial conditions
        in three-dimensional scenarios.

    Returns
    -------
    list of Path
        Paths to all generated CSV files.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined.")

    scenario = SCENARIOS[scenario_name]

    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    run_str = now.strftime("%Y%m%d_%H%M%S_%f")

    export_dir = EXPORT_ROOT / scenario_name / date_str / run_str
    export_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    csv_paths: List[Path] = []

    for i in range(n_simulations):
        seed = int(rng.integers(0, 2**32 - 1))

        ic_type = "l1_cloud"
        if scenario.dim == 3 and rng.random() < halo_fraction:
            ic_type = "halo_seed"

        df, termination_reason = simulate_single_run(
            scenario=scenario,
            steps=steps_per_sim,
            seed=seed,
            ic_type=ic_type,
        )

        if df.empty:
            continue

        filename = f"traj_{i:03d}_{ic_type}_{termination_reason}.csv"
        path = export_dir / filename
        df.to_csv(path, index=False)

        csv_paths.append(path)

    return csv_paths


# ======================================================================
# 2) Halo-Reference Export (Reference-Orbit Mode)
# ======================================================================
def simulate_halo_reference_run(
    scenario: ScenarioConfig,
    steps: int = 600,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate a CR3BP trajectory in halo-reference mode.

    The environment is configured with ``use_reference_orbit=True``
    and zero control input. The initial state is taken from a random
    point on the stored halo orbit.

    Parameters
    ----------
    scenario:
        Scenario configuration used to construct the environment.
    steps:
        Maximum number of integration steps before truncation.
    seed:
        Optional environment seed to randomize the reference index.

    Returns
    -------
    pandas.DataFrame
        Trajectory with columns
        ``t, x, y, z, vx, vy, vz, ax, ay, az``. An empty dataframe
        is returned if no valid samples could be generated.
    """
    env = Cr3bpStationKeepingEnv(
        scenario=scenario,
        max_steps=steps,
        integrator="DOP853",
        seed=seed,
        use_reference_orbit=True,
    )

    obs, info = env.reset()
    dim = env.dim
    dt = env.dt

    records: list[Dict[str, float]] = []

    sat = env.system.bodies[0]
    row0 = {
        "t": 0.0,
        "x": float(sat.position[0]),
        "y": float(sat.position[1]),
        "z": float(sat.position[2]) if dim == 3 else 0.0,
        "vx": float(sat.velocity[0]),
        "vy": float(sat.velocity[1]),
        "vz": float(sat.velocity[2]) if dim == 3 else 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
    }
    records.append(row0)

    for k in range(1, steps + 1):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]
        pos = rel_pos + env.target

        if env.use_reference_orbit and hasattr(env, "vel_ref_current"):
            vel = rel_vel + env.vel_ref_current
        else:
            vel = rel_vel

        acc = info.get("acc", np.zeros(dim, dtype=float))

        row = {
            "t": float(k * dt),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]) if dim == 3 else 0.0,
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]) if dim == 3 else 0.0,
            "ax": float(acc[0]),
            "ay": float(acc[1]),
            "az": float(acc[2]) if dim == 3 else 0.0,
        }
        records.append(row)

        if terminated or truncated:
            break

    env.close()

    if len(records) < 2:
        return pd.DataFrame()

    return pd.DataFrame(records)


def export_halo_reference_batch(
    scenario_name: str = "earth-moon-L1-3D",
    n_simulations: int = 1000,
    steps_per_sim: int = 600,
) -> List[Path]:
    """
    Generate multiple halo-reference trajectories and export as CSV.

    Trajectories are generated in reference-orbit mode with zero
    control. The filenames encode the initial-condition label
    ``"halo_ref_v1"`` and the termination reason ``"max_steps"``.

    Parameters
    ----------
    scenario_name:
        Key into the ``SCENARIOS`` registry, usually
        ``"earth-moon-L1-3D"`` for halo orbits around L1.
    n_simulations:
        Number of independent trajectories to generate.
    steps_per_sim:
        Maximum number of integration steps per trajectory.

    Returns
    -------
    list of Path
        Paths to all generated CSV files.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined.")

    scenario = SCENARIOS[scenario_name]

    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    run_str = now.strftime("%Y%m%d_%H%M%S_%f")

    export_dir = EXPORT_ROOT / f"{scenario_name}_halo_ref" / date_str / run_str
    export_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    paths: List[Path] = []

    for i in range(n_simulations):
        seed = int(rng.integers(0, 2**32 - 1))

        df = simulate_halo_reference_run(
            scenario=scenario,
            steps=steps_per_sim,
            seed=seed,
        )

        if df.empty:
            continue

        filename = f"traj_{i:03d}_halo_ref_v1_max_steps.csv"
        path = export_dir / filename
        df.to_csv(path, index=False)
        paths.append(path)

    return paths


# ======================================================================
# 3) High-Level Entry Point for Airflow
# ======================================================================
def generate_batch_simulations() -> List[Path]:
    """
    Generate all configured CR3BP trajectory batches.

    This function is intended as the primary entry point for
    orchestration systems such as Apache Airflow. It can be extended
    to trigger multiple export profiles (for example, local L1 cloud
    runs and halo-reference runs) within a single call.

    Returns
    -------
    list of Path
        Concatenated list of all generated CSV file paths.
    """
    paths_standard = export_batch(
        scenario_name="earth-moon-L1-3D",
        n_simulations=3000,
        steps_per_sim=600,
        halo_fraction=0.5,
    )

    paths_halo_ref = export_halo_reference_batch(
        scenario_name="earth-moon-L1-3D",
        n_simulations=1000,
        steps_per_sim=600,
    )

    return paths_standard + paths_halo_ref


if __name__ == "__main__":
    paths = generate_batch_simulations()
    print(f"Generated {len(paths)} trajectories in total.")
    for p in paths:
        print("  ->", p)
