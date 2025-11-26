from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv


EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def simulate_single_run(
    scenario: ScenarioConfig,
    steps: int = 3000,
    seed: int | None = None,
    ic_type: str = "l1_cloud",
    escape_radius: float = 3.0,
) -> pd.DataFrame:
    """
    Run a physics-only CR3BP simulation without control actions.

    The environment is configured with a chosen integrator and an
    initial-condition type. The rollout terminates on crashes,
    environment truncation, large excursions from the target region
    or non-finite states. The final crash or escape frame is not
    stored in the returned trajectory.
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

    # t = 0 initial state
    sat = env.system.bodies[0]
    initial_row = {
        "t": 0.0,
        "x": float(sat.position[0]),
        "y": float(sat.position[1]),
        "z": float(sat.position[2]) if dim == 3 else 0.0,
        "vx": float(sat.velocity[0]),
        "vy": float(sat.velocity[1]),
        "vz": float(sat.velocity[2]) if dim == 3 else 0.0,
        # Initial acceleration is set to zero as a harmless approximation
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
    }
    records.append(initial_row)

    # Main integration loop
    for k in range(1, steps + 1):
        action = np.zeros(dim, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]

        radius = float(np.linalg.norm(rel_pos))
        escape = radius > escape_radius

        # Do not store crash or escape frames
        if terminated or escape:
            break

        pos = rel_pos + env.target
        vel = rel_vel
        acc = info.get("acc", np.zeros(dim, dtype=float))

        t = k * dt

        row = {
            "t": float(t),
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

        # Numeric robustness: discard broken last frame
        non_finite_state = (
            not np.isfinite(pos).all()
            or not np.isfinite(vel).all()
            or not np.isfinite(acc).all()
        )
        if non_finite_state:
            records.pop()
            break

        if truncated:
            break

    env.close()

    # Discard trajectories that are too short to be useful
    if len(records) < 2:
        return pd.DataFrame()

    return pd.DataFrame(records)


def export_batch(
    scenario_name: str = "earth-moon-L1-3D",
    n_simulations: int = 300,
    steps_per_sim: int = 6000,
    halo_fraction: float = 0.3,
) -> List[Path]:
    """
    Generate multiple CR3BP trajectories and save them to CSV.

    A fraction of the simulations uses halo-like initial conditions for
    additional 3D variation; the remaining share uses L1-cloud initial
    conditions.
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
        if scenario.dim == 3:
            if rng.random() < halo_fraction:
                ic_type = "halo_seed"

        df = simulate_single_run(
            scenario=scenario,
            steps=steps_per_sim,
            seed=seed,
            ic_type=ic_type,
        )

        filename = f"traj_{i:03d}_{ic_type}.csv"
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
