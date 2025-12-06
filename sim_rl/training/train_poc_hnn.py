"""
Training script for the HNN-accelerated CR3BP environment.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

# Force single-threaded execution for determinism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONHASHSEED"] = "42"

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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

# Import the NEW HNN Environment
from sim_rl.cr3bp_hnn.env_cr3bp_station_keeping_hnn import Cr3bpStationKeepingEnvHNN
from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig

# Constants for logging
from sim_rl.cr3bp.constants import (
    SCALE_POS, SCALE_VEL,
    W_POS_HNN, W_VEL_HNN, W_CTRL_HNN, W_PLANAR_HNN
)

GLOBAL_SEED = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    print(f"[INFO] Global seed set to {seed}")

set_global_seed(GLOBAL_SEED)

TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 4
BASE_RUN_DIR = Path(__file__).resolve().parent / "runs_hnn" # Separate folder for HNN runs

def make_run_dirs(scenario: ScenarioConfig) -> dict[str, Path]:
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

def save_hnn_run_config(config_dir: Path, scenario: ScenarioConfig, model: PPO):
    cfg = {
        "type": "HNN_ACCELERATED",
        "global_seed": GLOBAL_SEED,
        "scenario": {
            "name": scenario.name,
            "system": scenario.system,
            "lagrange_point": scenario.lagrange_point,
            "dim": scenario.dim,
            "pos_noise": scenario.pos_noise,
            "vel_noise": scenario.vel_noise,
        },
        "scaling_constants": {
            "SCALE_POS": SCALE_POS,
            "SCALE_VEL": SCALE_VEL,
        },
        "hnn_rewards": {
            "W_POS_HNN": W_POS_HNN,
            "W_VEL_HNN": W_VEL_HNN,
            "W_CTRL_HNN": W_CTRL_HNN,
            "W_PLANAR_HNN": W_PLANAR_HNN,
        },
        "ppo_hyperparams": {
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "device": str(model.device),
        }
    }
    try:
        with (config_dir / "run_config.json").open("w") as f:
            json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not save config json: {e}")

class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training (HNN)", unit="step")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

def make_env_fn(scenario: ScenarioConfig, rank: int, use_reference_orbit: bool):
    def _thunk():
        env_seed = GLOBAL_SEED + rank
        env = Cr3bpStationKeepingEnvHNN( # NEW HNN CLASS
            scenario=scenario,
            use_reference_orbit=use_reference_orbit,
            seed=env_seed
        )
        env.reset(seed=env_seed)
        env = Monitor(env)
        return env
    return _thunk

def train_hnn_agent(
    scenario_name: str = "earth-moon-L1-3D",
    use_reference_orbit: bool = True,
):
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = SCENARIOS[scenario_name]
    print(f"[INFO] Starting HNN-Accelerated Training for: {scenario.name}")

    dirs = make_run_dirs(scenario)
    
    env = DummyVecEnv([
        make_env_fn(scenario, i, use_reference_orbit) for i in range(N_ENVS)
    ])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        seed=GLOBAL_SEED,
        tensorboard_log=str(dirs["logs"] / "tb")
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=str(dirs["models"]),
        name_prefix="ckpt_hnn"
    )
    
    eval_env = Cr3bpStationKeepingEnvHNN(
        scenario=scenario, 
        use_reference_orbit=use_reference_orbit,
        seed=GLOBAL_SEED + 100
    )
    eval_env = Monitor(eval_env, filename=str(dirs["logs"] / "eval.csv"))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(dirs["models"]),
        log_path=str(dirs["logs"]),
        eval_freq=20_000 // N_ENVS,
        deterministic=True,
        render=False
    )
    
    progress_callback = TqdmProgressCallback(total_timesteps=TOTAL_TIMESTEPS)

    print("[INFO] Model learning started...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, progress_callback]
    )
    
    final_path = dirs["models"] / "final_model_hnn.zip"
    model.save(str(final_path))
    print(f"[INFO] Training finished. Model saved to {final_path}")
    
    save_hnn_run_config(dirs["config"], scenario, model)

if __name__ == "__main__":
    train_hnn_agent(
        scenario_name="earth-moon-L1-3D",
        use_reference_orbit=True 
    )