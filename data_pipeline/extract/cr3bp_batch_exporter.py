# data_pipeline/extract/cr3bp_batch_exporter.py
"""
Batch exporter for generating CR3BP trajectories for HNN training.

This module produces multiple physics-only CR3BP simulations and stores
their trajectories as CSV files. The simulations are based on the core
CR3BP integrator from sim_rl and contain only (t, x, y, z, vx, vy, vz)
without any reinforcement-learning information.

Intended to be scheduled daily through an Airflow DAG.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from sim_rl.cr3bp.scenarios import ScenarioConfig, SCENARIOS
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv


# Root directory for exported raw CSV trajectories
EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def simulate_single_run(
    scenario: ScenarioConfig,
    steps: int = 3000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Run a physics-only CR3BP simulation using the base environment
    without applying any control actions.

    Parameters
    ----------
    scenario : ScenarioConfig
        The CR3BP configuration (system, L-point, dimension).
    steps : int
        Number of simulation steps to perform.
    seed : int or None
        Random seed for initial state perturbations.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the trajectory with columns:
        [t, x, y, z, vx, vy, vz]
    """
    env = Cr3bpStationKeepingEnv(scenario=scenario, seed=seed)

    obs, _ = env.reset()
    dim = env.dim
    dt = env.dt

    records = []

    for k in range(steps):
        # No control input → zero delta-v
        action = np.zeros(dim, dtype=np.float32)

        obs, reward, done, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]

        pos = rel_pos + env.target
        vel = rel_vel

        t = k * dt

        row = {
            "t": float(t),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]) if dim == 3 else 0.0,
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]) if dim == 3 else 0.0,
        }

        records.append(row)

        if done or truncated:
            break

    env.close()

    return pd.DataFrame(records)


def export_batch(
    scenario_name: str = "earth-moon-L1-3D",
    n_simulations: int = 20,
    steps_per_sim: int = 3000,
) -> list[Path]:
    """
    Generate multiple CR3BP physics trajectories and save them to CSV.

    Parameters
    ----------
    scenario_name : str
        Which scenario from SCENARIOS to use.
    n_simulations : int
        Number of simulations to generate.
    steps_per_sim : int
        Steps per trajectory.

    Returns
    -------
    list[Path]
        List of file paths of generated CSV files.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined.")

    scenario = SCENARIOS[scenario_name]

    export_dir = EXPORT_ROOT / scenario_name / datetime.now().strftime("%Y%m%d")
    export_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []

    for i in range(n_simulations):
        seed = np.random.randint(0, 2**32 - 1)
        df = simulate_single_run(scenario, steps=steps_per_sim, seed=seed)

        filename = f"traj_{i:03d}.csv"
        path = export_dir / filename
        df.to_csv(path, index=False)

        csv_paths.append(path)

    return csv_paths


if __name__ == "__main__":
    print("Generating CR3BP batch...")
    paths = export_batch()
    print(f"Generated {len(paths)} trajectories:")
    for p in paths:
        print("  →", p)
