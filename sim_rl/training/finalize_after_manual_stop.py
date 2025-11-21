"""
Finalize a CR3BP station-keeping run after manual training stop.

Typical usage from the project root:

.. code-block:: bash

    python -m sim_rl.training.finalize_after_manual_stop
    python -m sim_rl.training.finalize_after_manual_stop earth-moon-L1-3D

The script performs:

- Locate the latest run directory for a given scenario.
- Load the best available PPO model (``best_model.zip`` or last checkpoint).
- Save a copy as ``ppo_last.zip``.
- Export evaluation results from ``evaluations.npz`` to ``eval_rewards.csv``.
- Export a final rollout as CSV
  (``final_rollout_after_manual_stop.csv`` in the ``rollouts`` folder).
- Write ``run_config.json`` with scenario metadata and PPO hyperparameters.
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

# Runs are stored in sim_rl/training/runs relative to this file
BASE_RUN_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_SCENARIO_NAME = "earth-moon-L1-3D"


# ---------------------------------------------------------------------
# Helper functions (evaluation export, config export, rollout)
# ---------------------------------------------------------------------


def export_eval_to_csv(eval_log_dir: Path, csv_path: Path) -> None:
    """
    Export evaluation results from ``evaluations.npz`` to a CSV file.

    Parameters
    ----------
    eval_log_dir:
        Directory containing ``evaluations.npz`` produced by
        :class:`stable_baselines3.common.callbacks.EvalCallback`.
    csv_path:
        Output CSV path.
    """
    npz_path = eval_log_dir / "evaluations.npz"
    if not npz_path.exists():
        print(
            f"[WARN] No evaluations.npz found in {eval_log_dir}. "
            "Skipping evaluation CSV export."
        )
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]  # shape (n_evals,)
    results = data["results"]      # shape (n_evals, n_envs) or (n_evals,)
    ep_lengths = data.get("ep_lengths", None)

    if results.ndim == 2:
        mean_rewards = results.mean(axis=1)
    else:
        mean_rewards = results

    df = pd.DataFrame({"timesteps": timesteps, "mean_reward": mean_rewards})

    if ep_lengths is not None:
        if ep_lengths.ndim == 2:
            mean_lengths = ep_lengths.mean(axis=1)
        else:
            mean_lengths = ep_lengths
        df["mean_ep_length"] = mean_lengths

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Evaluation CSV written to: {csv_path}")


def _safe_float(x):
    """
    Try to cast ``x`` to float, returning ``None`` if this is not possible.

    This is mainly used to serialize certain Stable-Baselines3 schedule
    objects which are not directly JSON-serializable.
    """
    try:
        return float(x)
    except TypeError:
        return None


def save_run_config(config_dir: Path, scenario: ScenarioConfig, model: PPO) -> None:
    """
    Save scenario configuration and PPO hyperparameters as JSON.

    Parameters
    ----------
    config_dir:
        Target directory for ``run_config.json``.
    scenario:
        Scenario metadata.
    model:
        PPO model instance to read hyperparameters from.
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
    print(f"[INFO] Run configuration written to: {cfg_path}")


def rollout_policy_to_csv(
    scenario: ScenarioConfig,
    model: PPO,
    csv_path: Path,
    max_steps: int | None = None,
    deterministic: bool = True,
) -> None:
    """
    Run a rollout with the given policy and export it as CSV.

    Columns
    -------
    t, reward,
    x0, x1, (x2),
    v0, v1, (v2),
    dv0, dv1, (dv2)

    Parameters
    ----------
    scenario:
        Scenario used for environment construction.
    model:
        PPO policy to evaluate.
    csv_path:
        Output CSV path.
    max_steps:
        Optional maximum number of steps to simulate.
    deterministic:
        Whether to use deterministic actions.
    """
    env = Cr3bpStationKeepingEnv(scenario=scenario)

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0

    records = []

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

        row = {"t": t, "reward": reward}

        for i in range(dim):
            row[f"x{i}"] = pos[i]
        for i in range(dim):
            row[f"v{i}"] = vel[i]
        for i in range(dim):
            row[f"dv{i}"] = dv[i]

        records.append(row)
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Final rollout written to: {csv_path}")

    env.close()


# ---------------------------------------------------------------------
# Run directory resolution
# ---------------------------------------------------------------------


def find_latest_run_dir(scenario_name: str) -> Path:
    """
    Locate the latest run directory for a given scenario.

    First attempts to read ``latest_run.txt``. If this is not available,
    falls back to the lexicographically last ``run_*`` directory.

    Parameters
    ----------
    scenario_name:
        Scenario identifier (for example ``"earth-moon-L1-3D"``).

    Returns
    -------
    pathlib.Path
        Path to the latest run directory.
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
        else:
            print(
                f"[WARN] Path in latest_run.txt does not exist: {run_dir}. "
                "Falling back to last run_* directory."
            )

    candidates = [
        d for d in scenario_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run_* directories found in {scenario_root}"
        )

    latest_run = sorted(candidates)[-1]
    print(f"[INFO] Using latest run_* directory: {latest_run}")
    return latest_run


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------


def main() -> None:
    """
    Entry point for command-line usage.
    """
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    else:
        scenario_name = DEFAULT_SCENARIO_NAME

    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available scenarios: {list(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[scenario_name]
    print(f"[INFO] Finalizing run for scenario: {scenario.name}")

    run_dir = find_latest_run_dir(scenario.name)

    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    rollouts_dir = run_dir / "rollouts"
    config_dir = run_dir / "config"

    print(f"[INFO] Run directory: {run_dir}")

    best_model_path = models_dir / "best_model.zip"
    if not best_model_path.exists():
        checkpoint_candidates = sorted(models_dir.glob("checkpoint_*.zip"))
        if not checkpoint_candidates:
            raise FileNotFoundError(
                f"Neither best_model.zip nor checkpoint_*.zip found in {models_dir}"
            )
        best_model_path = checkpoint_candidates[-1]
        print(
            "[WARN] best_model.zip not found. "
            f"Using last checkpoint instead: {best_model_path}"
        )
    else:
        print(f"[INFO] Using best_model: {best_model_path}")

    env_for_load = Cr3bpStationKeepingEnv(scenario=scenario)
    model = PPO.load(best_model_path, env=env_for_load, device="cpu")
    print("[INFO] Model loaded.")

    ppo_last_path = models_dir / "ppo_last.zip"
    model.save(ppo_last_path)
    print(f"[INFO] ppo_last.zip written to: {ppo_last_path}")

    eval_csv_path = logs_dir / "eval_rewards.csv"
    export_eval_to_csv(logs_dir, eval_csv_path)

    save_run_config(config_dir, scenario, model)

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
