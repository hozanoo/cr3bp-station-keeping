"""Finalize a CR3BP PPO training run after manual interruption.

Usage
-----

From the project root (where ``sim_rl`` lives), run:

.. code-block:: bash

   python -m sim_rl.training.finalize_after_manual_stop
   python -m sim_rl.training.finalize_after_manual_stop earth-moon-L1-3D

Purpose
-------

This script performs a standardized post-processing step for a given
scenario:

- locate the latest run directory for the scenario,
- load ``best_model.zip`` (or the last checkpoint as fallback),
- save a consolidated ``ppo_last.zip``,
- export evaluation statistics from ``evaluations.npz`` to
  ``eval_rewards.csv`` (if available),
- record a final rollout as CSV,
- store ``run_config.json`` containing scenario and PPO parameters.

Prerequisites
-------------

- Stable-Baselines3 installed,
- project structure:

  ``sim_rl/training/runs/<scenario_name>/run_YYYYMMDD_HHMMSS/...``

- at least one trained run with models and logs for the scenario.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv
from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig

# Base directory for runs (same convention as in train_poc.py)
BASE_RUN_DIR = Path(__file__).parent / "runs"

#: Default scenario name used when none is provided via CLI.
DEFAULT_SCENARIO_NAME = "earth-moon-L1-3D"


# ---------------------------------------------------------------------
# Helper functions (eval export, run config, rollout)
# ---------------------------------------------------------------------


def export_eval_to_csv(eval_log_dir: Path, csv_path: Path) -> None:
    """Export evaluation results from ``evaluations.npz`` to CSV.

    This expects the file produced by :class:`stable_baselines3.common.callbacks.EvalCallback`
    and writes a CSV file containing timesteps, mean episode reward and
    (if available) mean episode length.

    Parameters
    ----------
    eval_log_dir : pathlib.Path
        Directory containing ``evaluations.npz``.
    csv_path : pathlib.Path
        Output path for the CSV file.
    """
    npz_path = eval_log_dir / "evaluations.npz"
    if not npz_path.exists():
        print(
            f"[WARN] No evaluations.npz found in {eval_log_dir}. "
            "Skipping eval CSV export."
        )
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]  # shape (n_evals,)
    results = data["results"]  # shape (n_evals, n_envs) or (n_evals,)
    ep_lengths = data.get("ep_lengths", None)

    # Average across envs if necessary
    if results.ndim == 2:
        mean_rewards = results.mean(axis=1)
    else:
        mean_rewards = results

    df = pd.DataFrame(
        {
            "timesteps": timesteps,
            "mean_reward": mean_rewards,
        }
    )

    if ep_lengths is not None:
        if ep_lengths.ndim == 2:
            mean_lengths = ep_lengths.mean(axis=1)
        else:
            mean_lengths = ep_lengths
        df["mean_ep_length"] = mean_lengths

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Evaluation CSV saved to: {csv_path}")


def _safe_float(x):
    """Try to safely cast a value to float.

    This is mainly used to convert Schedule-like objects from
    Stable-Baselines3 (e.g. ``clip_range``) into simple floats
    for run configuration logging.

    Parameters
    ----------
    x :
        Value to be converted.

    Returns
    -------
    float or None
        Float value if conversion is possible, otherwise ``None``.
    """
    try:
        return float(x)
    except TypeError:
        return None


def save_run_config(config_dir: Path, scenario: ScenarioConfig, model: PPO) -> None:
    """Save scenario configuration and PPO hyperparameters as JSON.

    Parameters
    ----------
    config_dir : pathlib.Path
        Directory where the configuration file should be written.
    scenario : ScenarioConfig
        Scenario used for this training run.
    model : stable_baselines3.PPO
        PPO model from which to read hyperparameters.
    """
    cfg = {
        "scenario": {
            "name": scenario.name,
            "system": scenario.system,
            "lagrange_point": scenario.lagrange_point,
            "dim": scenario.dim,
            "action_mode": scenario.action_mode,
            "pos_noise": scenario.pos_noise,
            "vel_noise": scenario.vel_noise,
        },
        "ppo": {
            "learning_rate": _safe_float(model.learning_rate),
            "gamma": _safe_float(model.gamma),
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "gae_lambda": _safe_float(model.gae_lambda),
            "clip_range": _safe_float(model.clip_range),
            "ent_coef": _safe_float(model.ent_coef),
            "vf_coef": _safe_float(model.vf_coef),
            "max_grad_norm": _safe_float(model.max_grad_norm),
            "policy": str(model.policy.__class__.__name__),
            "device": str(model.device),
        },
    }

    cfg_path = config_dir / "run_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[INFO] Run configuration saved to: {cfg_path}")


def rollout_policy_to_csv(
    scenario: ScenarioConfig,
    model: PPO,
    csv_path: Path,
    max_steps: int | None = None,
    deterministic: bool = True,
) -> None:
    """Run a single rollout and store state and Δv history as CSV.

    The CSV columns follow the scheme:

    - ``t`` (time),
    - ``reward``,
    - ``x0, x1, (x2)`` – position components,
    - ``v0, v1, (v2)`` – velocity components,
    - ``dv0, dv1, (dv2)`` – applied delta-v.

    Parameters
    ----------
    scenario : ScenarioConfig
        Scenario in which the rollout is executed.
    model : stable_baselines3.PPO
        Policy used to generate actions.
    csv_path : pathlib.Path
        Output CSV path.
    max_steps : int or None, optional
        Optional maximum number of steps for the rollout. If ``None``,
        the rollout continues until the environment terminates.
    deterministic : bool, optional
        Whether to use deterministic actions during rollout.
    """
    env = Cr3bpStationKeepingEnv(scenario=scenario)

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0

    records: list[dict] = []

    dt = env.dt
    dim = env.dim

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]

        pos = rel_pos + env.target
        vel = rel_vel
        dv = info.get("dv", np.zeros(dim))

        t = step * dt

        row: dict[str, float] = {"t": t, "reward": reward}
        for i in range(dim):
            row[f"x{i}"] = float(pos[i])
        for i in range(dim):
            row[f"v{i}"] = float(vel[i])
        for i in range(dim):
            row[f"dv{i}"] = float(dv[i])

        records.append(row)
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Final rollout saved to: {csv_path}")

    env.close()


# ---------------------------------------------------------------------
# Run directory discovery
# ---------------------------------------------------------------------


def find_latest_run_dir(scenario_name: str) -> Path:
    """Find the latest run directory for a given scenario.

    The function first attempts to read ``latest_run.txt`` under the
    scenario root. If that fails or points to a non-existing directory,
    it falls back to selecting the lexicographically last ``run_*``
    directory, which is consistent with the timestamp-based naming.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario (e.g. ``"earth-moon-L1-3D"``).

    Returns
    -------
    pathlib.Path
        Path to the latest run directory.

    Raises
    ------
    FileNotFoundError
        If the scenario root or any run directory cannot be found.
    """
    scenario_root = BASE_RUN_DIR / scenario_name
    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario root {scenario_root} does not exist.")

    latest_txt = scenario_root / "latest_run.txt"
    if latest_txt.exists():
        content = latest_txt.read_text(encoding="utf-8").strip()
        run_dir = Path(content)
        if run_dir.exists():
            print(f"[INFO] Using run from latest_run.txt: {run_dir}")
            return run_dir
        print(
            f"[WARN] Path in latest_run.txt does not exist: {run_dir}, "
            "falling back to last run_* directory."
        )

    # Fallback: select the last run_* directory
    candidates = [
        d
        for d in scenario_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run_* directories found in {scenario_root}."
        )

    latest_run = sorted(candidates)[-1]
    print(f"[INFO] Using last run_* directory: {latest_run}")
    return latest_run


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------


def main() -> None:
    """Finalize the latest run of a scenario after manual stopping.

    This function:

    - selects the scenario (from CLI or default),
    - locates the latest run directory,
    - loads the best model (or last checkpoint),
    - saves a consolidated ``ppo_last.zip``,
    - exports evaluation statistics to CSV,
    - stores a JSON run configuration,
    - performs a final rollout and saves it as CSV.
    """
    # Scenario selection from CLI
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    else:
        scenario_name = DEFAULT_SCENARIO_NAME

    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. "
            f"Known scenarios: {list(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[scenario_name]
    print(f"[INFO] Finalizing run for scenario: {scenario.name}")

    # Locate latest run directory
    run_dir = find_latest_run_dir(scenario.name)

    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    rollouts_dir = run_dir / "rollouts"
    config_dir = run_dir / "config"

    print(f"[INFO] Run directory: {run_dir}")

    # Locate best_model.zip
    best_model_path = models_dir / "best_model.zip"
    if not best_model_path.exists():
        # Fallback: last checkpoint
        checkpoint_candidates = sorted(models_dir.glob("checkpoint_*.zip"))
        if not checkpoint_candidates:
            raise FileNotFoundError(
                f"Neither best_model.zip nor checkpoint_*.zip found in {models_dir}."
            )
        best_model_path = checkpoint_candidates[-1]
        print(
            "[WARN] No best_model.zip found. "
            f"Using last checkpoint instead: {best_model_path}"
        )
    else:
        print(f"[INFO] Using best_model: {best_model_path}")

    # Load model on CPU
    env_for_load = Cr3bpStationKeepingEnv(scenario=scenario)
    model = PPO.load(best_model_path, env=env_for_load, device="cpu")
    print("[INFO] Model loaded.")

    # Save ppo_last.zip
    ppo_last_path = models_dir / "ppo_last.zip"
    model.save(ppo_last_path)
    print(f"[INFO] ppo_last.zip saved to: {ppo_last_path}")

    # Export evaluation CSV if evaluations.npz exists
    eval_csv_path = logs_dir / "eval_rewards.csv"
    export_eval_to_csv(logs_dir, eval_csv_path)

    # Save or update run configuration
    save_run_config(config_dir, scenario, model)

    # Final rollout
    final_rollout_path = rollouts_dir / "final_rollout_after_manual_stop.csv"
    rollout_policy_to_csv(
        scenario=scenario,
        model=model,
        csv_path=final_rollout_path,
        max_steps=None,
        deterministic=True,
    )

    print("[INFO] Finalization completed.")


if __name__ == "__main__":
    main()
