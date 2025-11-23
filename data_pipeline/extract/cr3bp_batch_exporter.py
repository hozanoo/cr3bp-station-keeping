# data_pipeline/extract/cr3bp_batch_exporter.py
"""
Batch exporter for generating CR3BP trajectories for HNN training.

This module produces multiple physics-only CR3BP simulations and stores
their trajectories as CSV files. The simulations are based on the core
CR3BP integrator from sim_rl and contain only (t, x, y, z, vx, vy, vz)
without any reinforcement-learning information.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv


# Root directory for exported raw CSV trajectories
EXPORT_ROOT = Path(__file__).resolve().parent.parent / "raw_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def simulate_single_run(
    scenario: ScenarioConfig,
    steps: int = 3000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run a physics-only CR3BP simulation without control actions."""
    env = Cr3bpStationKeepingEnv(scenario=scenario, seed=seed)

    obs, _ = env.reset()
    dim = env.dim
    dt = env.dt

    records = []

    for k in range(steps):
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
    n_simulations: int = 300,
    steps_per_sim: int = 6000,
) -> List[Path]:
    """Generate multiple CR3BP trajectories and save them to CSV."""
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Scenario {scenario_name!r} not defined.")

    scenario = SCENARIOS[scenario_name]

    # Zeitstempel: Datum + eindeutiger Run-Ordner pro Batch
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    run_str = now.strftime("%Y%m%d_%H%M%S_%f")

    export_dir = EXPORT_ROOT / scenario_name / date_str / run_str
    export_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []

    for i in range(n_simulations):
        seed = int(np.random.randint(0, 2**32 - 1))
        df = simulate_single_run(scenario, steps=steps_per_sim, seed=seed)

        filename = f"traj_{i:03d}.csv"
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
