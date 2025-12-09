"""
Determinist training script for the CR3BP station-keeping task (Repo Version).

This script enforces strict reproducibility by:
1. Disabling GPU and forcing single-threaded CPU execution.
2. Setting fixed seeds for Python, NumPy, and PyTorch.
3. Using the deterministic 'Repo' environment with hard-coded scaling.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

# ======================================================================
# 1. HARD DETERMINISM ENFORCEMENT (Must be before other imports)
# ======================================================================

# Force single-threaded execution to avoid race conditions in math libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Disable CUDA / GPU completely
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set a fixed hash seed for Python dictionaries
os.environ["PYTHONHASHSEED"] = "42"


# ======================================================================
# Imports
# ======================================================================

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)

# Import the deterministic Repo environment
from sim_rl.cr3bp.env_cr3bp_station_keeping_repo import Cr3bpStationKeepingEnvRepo
from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig

# Import constants to log them for reproducibility
from sim_rl.cr3bp.constants import (
    SCALE_POS, SCALE_VEL,
    W_POS_REPO, W_VEL_REPO, W_CTRL_REPO, W_PLANAR_REPO
)

# ======================================================================
# Global Seed Function
# ======================================================================

GLOBAL_SEED = 42

def set_global_seed(seed: int):
    """
    Set seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # PyTorch CPU threading limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"[INFO] Global seed set to {seed}. Deterministic algorithms enabled.")

# Apply seed immediately
set_global_seed(GLOBAL_SEED)


# ======================================================================
# Configuration & Constants
# ======================================================================

TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 4
BASE_RUN_DIR = Path(__file__).resolve().parent / "runs_repo"


# ======================================================================
# Helper Functions (Run Management & Logging)
# ======================================================================

def make_run_dirs(scenario: ScenarioConfig) -> dict[str, Path]:
    """
    Create directory structure: runs_repo/<scenario_name>/run_<timestamp>
    And writes 'latest_run.txt' for easy access.
    """
    scenario_root = BASE_RUN_DIR / scenario.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scenario_root / f"run_{timestamp}"

    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    config_dir = run_dir / "config"
    rollouts_dir = run_dir / "rollouts"

    for d in (models_dir, logs_dir, config_dir, rollouts_dir):
        d.mkdir(parents=True, exist_ok=True)

    latest_path = scenario_root / "latest_run.txt"
    try:
        latest_path.write_text(str(run_dir), encoding="utf-8")
    except OSError:
        pass

    return {
        "run_dir": run_dir,
        "models": models_dir,
        "logs": logs_dir,
        "config": config_dir,
        "rollouts": rollouts_dir,
    }

def export_eval_to_csv(eval_log_dir: Path, csv_path: Path) -> None:
    """
    Export evaluation results from ``evaluations.npz`` to a CSV file.
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
    print(f"[INFO] Evaluation CSV written to: {csv_path}")

def save_repo_config(config_dir: Path, scenario: ScenarioConfig, model: PPO):
    """
    Save the configuration including the specific Repo constants.
    """
    cfg = {
        "global_seed": GLOBAL_SEED,
        "scenario": {
            "name": scenario.name,
            "system": scenario.system,
            "lagrange_point": scenario.lagrange_point,
            "action_mode": scenario.action_mode,
            "dim": scenario.dim,
            "pos_noise": scenario.pos_noise,
            "vel_noise": scenario.vel_noise,
        },
        "scaling_constants": {
            "SCALE_POS": SCALE_POS,
            "SCALE_VEL": SCALE_VEL,
        },
        "repo_rewards": {
            "W_POS_REPO": W_POS_REPO,
            "W_VEL_REPO": W_VEL_REPO,
            "W_CTRL_REPO": W_CTRL_REPO,
            "W_PLANAR_REPO": W_PLANAR_REPO,
        },
        "ppo_hyperparams": {
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "seed": model.seed,
            "device": str(model.device),
        }
    }
    
    try:
        with (config_dir / "run_config.json").open("w") as f:
            json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not save config json: {e}")


# ======================================================================
# Helper: Rollout Export Function (for SimulationRecorder)
# ======================================================================

def rollout_policy_to_csv(
    scenario: ScenarioConfig,
    model: PPO,
    csv_path: Path,
    max_steps: int | None = None,
    deterministic: bool = True,
    use_reference_orbit: bool = False,
) -> None:
    """
    Run a rollout with the given policy and export it as CSV.
    Uses the Repo environment logic.
    """
    # Create a fresh environment for recording
    env = Cr3bpStationKeepingEnvRepo(scenario=scenario, use_reference_orbit=use_reference_orbit)
    
    # Reset
    obs, info = env.reset(seed=42) # Fixed seed for consistency
    done = False
    truncated = False
    step = 0

    records: list[dict[str, Any]] = []
    dt = env.dt
    dim = env.dim

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step env
        obs, reward, done, truncated, info = env.step(action)

        # In Repo Env, obs is scaled! 
        # For CSV logging we want physical values.
        
        # Get scaled relative state from obs
        scaled_pos = obs[:dim]
        scaled_vel = obs[dim : 2 * dim]
        
        # Unscale
        rel_pos = scaled_pos / SCALE_POS
        rel_vel = scaled_vel / SCALE_VEL

        pos = rel_pos + env.target
        
        # Velocity in rotating frame
        if use_reference_orbit and hasattr(env, "vel_ref_current"):
            vel = rel_vel + env.vel_ref_current
        else:
            vel = rel_vel

        dv = info.get("dv", np.zeros(dim))
        t = step * dt

        row: dict[str, Any] = {"t": t, "reward": float(reward)}

        for i in range(dim):
            row[f"pos_rot_abs_{i}"] = float(pos[i])
        for i in range(dim):
            row[f"vel_rot_abs_{i}"] = float(vel[i])
        for i in range(dim):
            row[f"dv_{i}"] = float(dv[i])

        records.append(row)
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    env.close()


# ======================================================================
# Callbacks
# ======================================================================

class TqdmProgressCallback(BaseCallback):
    """
    Simple progress bar.
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training (Repo)", unit="step")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


class SimulationRecorderCallback(BaseCallback):
    """
    Callback that records full simulations at selected rollout indices.
    Logic: Save first 20 rollouts, then every 50th rollout (50, 100, 150...).
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        rollouts_dir: Path,
        verbose: int = 0,
        max_steps: int | None = None,
        use_reference_orbit: bool = False,
    ):
        super().__init__(verbose)
        self.scenario = scenario
        self.rollouts_dir = rollouts_dir
        self.max_steps = max_steps
        self.use_reference_orbit = use_reference_orbit
        
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        
        should_record = False
        
        if self.rollout_count <= 20:
            should_record = True
        elif self.rollout_count % 50 == 0:
            should_record = True
            
        if not should_record:
            return

        # Save CSV
        csv_name = f"sim_{self.rollout_count:04d}_steps_{self.num_timesteps}.csv"
        csv_path = self.rollouts_dir / csv_name

        rollout_policy_to_csv(
            scenario=self.scenario,
            model=self.model,
            csv_path=csv_path,
            max_steps=self.max_steps,
            deterministic=True,
            use_reference_orbit=self.use_reference_orbit,
        )

    def _on_step(self) -> bool:
        return True


# ======================================================================
# Environment Factory
# ======================================================================

def make_env_fn(scenario: ScenarioConfig, rank: int, use_reference_orbit: bool, log_dir: Path):
    """
    Create a deterministic environment instance.
    Importantly, we pass a unique seed derived from GLOBAL_SEED and rank.
    Includes Monitor for CSV logging.
    """
    def _thunk():
        # Derive specific seed for this environment instance
        env_seed = GLOBAL_SEED + rank
        
        env = Cr3bpStationKeepingEnvRepo(
            scenario=scenario,
            use_reference_orbit=use_reference_orbit,
            seed=env_seed 
        )
        
        env.reset(seed=env_seed)
        
        # Save train logs to CSV
        filename = log_dir / f"train_monitor_{rank}.csv"
        env = Monitor(env, filename=str(filename))
        return env
    return _thunk


# ======================================================================
# Training Function
# ======================================================================

def train_repo(
    scenario_name: str = "earth-moon-L1-3D",
    use_reference_orbit: bool = True,
):
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = SCENARIOS[scenario_name]
    print(f"[INFO] Starting REPO Training for: {scenario.name}")
    print(f"[INFO] Reference Orbit Mode: {use_reference_orbit}")

    # Create directories
    dirs = make_run_dirs(scenario)
    
    # Create Vectorized Environment
    # N_ENVS = 4 allows for batch diversity while running on a single CPU thread sequentially.
    env = DummyVecEnv([
        make_env_fn(scenario, i, use_reference_orbit, dirs["logs"]) for i in range(N_ENVS)
    ])

    # Initialize PPO with strict CPU settings
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,  # Consistent updates
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",      # Strict CPU
        seed=GLOBAL_SEED,  # Pass seed to PPO
        tensorboard_log=str(dirs["logs"] / "tb")
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=str(dirs["models"]),
        name_prefix="ckpt_repo"
    )
    
    # Evaluation Callback
    # We use a separate evaluation environment with its own seed
    eval_env = Cr3bpStationKeepingEnvRepo(
        scenario=scenario, 
        use_reference_orbit=use_reference_orbit,
        seed=GLOBAL_SEED + 100
    )
    eval_env = Monitor(eval_env, filename=str(dirs["logs"] / "eval_monitor.csv"))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(dirs["models"]),
        log_path=str(dirs["logs"]),
        eval_freq=20_000 // N_ENVS,
        deterministic=True,
        render=False
    )
    
    progress_callback = TqdmProgressCallback(total_timesteps=TOTAL_TIMESTEPS)

    # Simulation Recorder
    sim_recorder = SimulationRecorderCallback(
        scenario=scenario,
        rollouts_dir=dirs["rollouts"],
        verbose=0,
        max_steps=None, 
        use_reference_orbit=use_reference_orbit
    )

    # Start Training
    print("[INFO] Model learning started...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, progress_callback, sim_recorder]
    )
    
    # Save final model
    final_path = dirs["models"] / "final_model.zip"
    model.save(str(final_path))
    print(f"[INFO] Training finished. Model saved to {final_path}")
    
    # Save Config
    save_repo_config(dirs["config"], scenario, model)

    # Export Eval Data to CSV
    export_eval_to_csv(dirs["logs"], dirs["logs"] / "eval_rewards.csv")


if __name__ == "__main__":
    # Ensure we run this from the module context if needed, but direct execution is fine
    # checks are done at the top of the file.
    train_repo(
        scenario_name="earth-moon-L1-3D",
        use_reference_orbit=True 
    )