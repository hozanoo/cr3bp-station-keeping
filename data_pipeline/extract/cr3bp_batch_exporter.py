from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv


EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def simulate_single_run(
    scenario: ScenarioConfig,
    steps: int = 300,
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

    Returns
    -------
    trajectory : pandas.DataFrame
        Columns: t, x, y, z, vx, vy, vz, ax, ay, az
    termination_reason : str
        One of: "max_steps", "escape", "crash_primary1",
        "crash_primary2", "nan", "truncated".
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

        # Determine termination reason BEFORE appending a broken state
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
    steps_per_sim: int = 300,
    halo_fraction: float = 0.5,
) -> List[Path]:
    """
    Generate multiple CR3BP trajectories and save them to CSV.

    A fraction of the simulations uses halo-like initial conditions for
    additional 3D variation; the remaining share uses L1-cloud initial
    conditions.

    The termination_reason is encoded in the filename, so that the
    database loader can reconstruct metadata without re-simulating.
    Example filename:

        traj_012_l1_cloud_max_steps.csv
        traj_013_halo_seed_escape.csv
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


def generate_batch_simulations() -> List[Path]:
    """Convenience wrapper used by the Airflow DAG."""
    return export_batch()


if __name__ == "__main__":
    paths = generate_batch_simulations()
    print(f"Generated {len(paths)} trajectories:")
    for p in paths:
        print("  ->", p)
