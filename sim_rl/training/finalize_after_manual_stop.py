"""
Finalize a training run after manual stop (Ctrl+C) or crash.
Supports Repo, Robust, and HNN modes.

Usage:
    python -m sim_rl.training.finalize_training --mode repo
    python -m sim_rl.training.finalize_training --mode robust
    python -m sim_rl.training.finalize_training --mode hnn --scenario earth-moon-L1-3D
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO

# Import all environment classes
from sim_rl.cr3bp.env_cr3bp_station_keeping_repo import Cr3bpStationKeepingEnvRepo
from sim_rl.cr3bp.env_cr3bp_station_keeping_robust import Cr3bpStationKeepingEnvRobust
from sim_rl.cr3bp_hnn.env_cr3bp_station_keeping_hnn import Cr3bpStationKeepingEnvHNN

from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig
# Import constants for config logging
from sim_rl.cr3bp.constants import (
    SCALE_POS, SCALE_VEL,
    W_POS_REPO, W_VEL_REPO, W_CTRL_REPO, W_PLANAR_REPO,
    W_POS_ROBUST, W_VEL_ROBUST, W_CTRL_ROBUST, W_PLANAR_ROBUST,
    W_POS_HNN, W_VEL_HNN, W_CTRL_HNN, W_PLANAR_HNN,
    HALO_DEADBAND
)

# Base path relative to this script
BASE_TRAINING_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Mode Configuration
# ---------------------------------------------------------------------

MODE_CONFIG = {
    "repo": {
        "dir": "runs_repo",
        "env_class": Cr3bpStationKeepingEnvRepo,
    },
    "robust": {
        "dir": "runs_robust",
        "env_class": Cr3bpStationKeepingEnvRobust,
    },
    "hnn": {
        "dir": "runs_hnn",
        "env_class": Cr3bpStationKeepingEnvHNN,
    },
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def export_eval_to_csv(eval_log_dir: Path, csv_path: Path) -> None:
    """
    Export evaluation results from evaluations.npz to a CSV file.
    """
    npz_path = eval_log_dir / "evaluations.npz"
    if not npz_path.exists():
        print(f"[WARN] No evaluations.npz found in {eval_log_dir}")
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
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


def save_run_config(config_dir: Path, scenario: ScenarioConfig, model: PPO, mode: str) -> None:
    """
    Save run configuration based on the selected mode.
    """
    # Base config
    cfg = {
        "mode": mode,
        "scenario": {
            "name": scenario.name,
            "system": scenario.system,
            "lagrange_point": scenario.lagrange_point,
            "dim": scenario.dim,
            "pos_noise": scenario.pos_noise,
            "vel_noise": scenario.vel_noise,
        },
        "ppo": {
            "learning_rate": float(model.learning_rate) if isinstance(model.learning_rate, float) else "scheduled",
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "device": str(model.device),
        },
        "scaling": {
            "SCALE_POS": SCALE_POS,
            "SCALE_VEL": SCALE_VEL
        }
    }

    # Mode specific rewards
    if mode == "repo":
        cfg["rewards"] = {
            "W_POS": W_POS_REPO, "W_VEL": W_VEL_REPO, 
            "W_CTRL": W_CTRL_REPO, "W_PLANAR": W_PLANAR_REPO,
            "DEADBAND": 0.0
        }
    elif mode == "robust":
        cfg["rewards"] = {
            "W_POS": W_POS_ROBUST, "W_VEL": W_VEL_ROBUST, 
            "W_CTRL": W_CTRL_ROBUST, "W_PLANAR": W_PLANAR_ROBUST,
            "DEADBAND": HALO_DEADBAND
        }
    elif mode == "hnn":
        cfg["rewards"] = {
            "W_POS": W_POS_HNN, "W_VEL": W_VEL_HNN, 
            "W_CTRL": W_CTRL_HNN, "W_PLANAR": W_PLANAR_HNN
        }

    cfg_path = config_dir / "run_config_finalized.json"
    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"[INFO] Config written to: {cfg_path}")
    except Exception as e:
        print(f"[WARN] Could not write config: {e}")


def rollout_policy_to_csv(
    EnvClass: Type[gym.Env],
    scenario: ScenarioConfig,
    model: PPO,
    csv_path: Path,
    use_reference_orbit: bool = True,
) -> None:
    """
    Run a rollout and export as CSV.
    """
    # Instantiate the specific environment class
    env = EnvClass(scenario=scenario, use_reference_orbit=use_reference_orbit)

    # Use fixed seed for comparison
    obs, info = env.reset(seed=42)
    done = False
    truncated = False
    step = 0
    records = []
    
    dt = env.dt
    dim = env.dim

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Reconstruct physical state logic (generic for all envs)
        # Note: In Robust/Repo, obs is scaled. We use env.target to reconstruct.
        # But for the CSV we want readable values.
        
        # We assume the env has 'target' and 'vel_ref_current' attributes
        # which is true for all our classes.
        
        # Unscale manually for logging if needed, or rely on internal env logic
        # Here we try to get physical state directly if possible, or reconstruct
        # For simplicity and robustness across classes, we reconstruct from obs using constants
        
        rel_pos = obs[:dim] / SCALE_POS
        rel_vel = obs[dim : 2*dim] / SCALE_VEL
        
        pos = rel_pos + env.target
        
        if hasattr(env, "vel_ref_current"):
            vel = rel_vel + env.vel_ref_current
        else:
            vel = rel_vel

        dv = info.get("dv", np.zeros(dim))
        t = step * dt

        row = {"t": t, "reward": float(reward)}
        for i in range(dim):
            row[f"pos_{i}"] = float(pos[i])
        for i in range(dim):
            row[f"vel_{i}"] = float(vel[i])
        for i in range(dim):
            row[f"dv_{i}"] = float(dv[i])

        records.append(row)
        step += 1

    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Final rollout written to: {csv_path}")
    env.close()


def find_latest_run_dir(base_dir: Path, scenario_name: str) -> Path:
    """
    Locate the latest run directory.
    """
    scenario_root = base_dir / scenario_name
    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario root {scenario_root} does not exist.")

    latest_txt = scenario_root / "latest_run.txt"
    if latest_txt.exists():
        try:
            content = latest_txt.read_text(encoding="utf-8").strip()
            run_dir = Path(content)
            if run_dir.exists():
                return run_dir
        except Exception:
            pass

    # Fallback: search manually
    candidates = [
        d for d in scenario_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {scenario_root}")

    return sorted(candidates)[-1]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Finalize training run manually.")
    parser.add_argument("--mode", type=str, required=True, choices=["repo", "robust", "hnn"],
                        help="Which training mode to finalize.")
    parser.add_argument("--scenario", type=str, default="earth-moon-L1-3D",
                        help="Scenario name.")
    args = parser.parse_args()

    # 1. Setup based on mode
    config = MODE_CONFIG[args.mode]
    base_run_dir = BASE_TRAINING_DIR / config["dir"]
    EnvClass = config["env_class"]
    scenario = SCENARIOS[args.scenario]

    print(f"[INFO] Finalizing {args.mode.upper()} run for {scenario.name}...")

    # 2. Find directory
    try:
        run_dir = find_latest_run_dir(base_run_dir, scenario.name)
        print(f"[INFO] Found run directory: {run_dir}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    rollouts_dir = run_dir / "rollouts"
    config_dir = run_dir / "config"

    # 3. Load Model (Best or Checkpoint)
    best_model_path = models_dir / "best_model.zip"
    if not best_model_path.exists():
        # Find latest checkpoint
        ckpts = sorted(models_dir.glob("ckpt_*.zip"))
        if not ckpts:
            # Fallback for old naming convention if present
            ckpts = sorted(models_dir.glob("checkpoint_*.zip"))
        
        if not ckpts:
            print("[ERROR] No models found in models directory.")
            return
        
        best_model_path = ckpts[-1]
        print(f"[WARN] 'best_model.zip' not found. Using latest checkpoint: {best_model_path.name}")
    else:
        print(f"[INFO] Loading {best_model_path.name}")

    # Load agent
    # We need a dummy env for loading to infer shapes if needed, but usually not for prediction
    # However, standard practice is to pass the env.
    dummy_env = EnvClass(scenario=scenario, use_reference_orbit=True)
    model = PPO.load(best_model_path, env=dummy_env, device="cpu")

    # 4. Export Eval CSV
    export_eval_to_csv(logs_dir, logs_dir / "eval_rewards.csv")

    # 5. Save Config
    save_run_config(config_dir, scenario, model, args.mode)

    # 6. Generate Final Rollout
    final_csv_path = rollouts_dir / "final_rollout_manual.csv"
    rollout_policy_to_csv(
        EnvClass=EnvClass,
        scenario=scenario,
        model=model,
        csv_path=final_csv_path,
        use_reference_orbit=True
    )

    print("[INFO] Finalization complete. All files are ready.")

if __name__ == "__main__":
    main()