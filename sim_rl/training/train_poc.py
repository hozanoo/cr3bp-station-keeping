"""
PPO training entry point for the CR3BP station-keeping environment.

This module trains a PPO agent on a selected scenario and manages
run directories, checkpoints, evaluation logs and rollout exports.

Typical usage from the project root:

.. code-block:: bash

    python -m sim_rl.training.train_poc
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import json
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)

from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv
from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig

from sim_rl.cr3bp.constants import (
    W_POS,
    W_VEL,
    W_CTRL,
    L1_DEADBAND,
    L1_FAR_LIMIT,
    CRASH_PENALTY,
    PLANAR_Z_THRESHOLD,
    PLANAR_VZ_THRESHOLD,
    W_PLANAR,
    W_POS_REF,
    W_VEL_REF,
    W_CTRL_REF,
    W_PLANAR_REF,
)

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------

TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 4

# Runs are stored in sim_rl/training/runs relative to this file
BASE_RUN_DIR = Path(__file__).resolve().parent / "runs"


# ======================================================================
# Run management
# ======================================================================


def make_run_dirs(scenario: ScenarioConfig) -> dict[str, Path]:
    """
    Create the run directory hierarchy for a given scenario.

    Structure
    ---------
    runs/<scenario.name>/run_YYYYMMDD_HHMMSS/

    Returns
    -------
    dict[str, Path]
        Mapping containing paths for ``run_dir``, ``models``, ``logs``,
        ``rollouts`` and ``config``.
    """
    scenario_root = BASE_RUN_DIR / scenario.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scenario_root / f"run_{timestamp}"

    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    rollouts_dir = run_dir / "rollouts"
    config_dir = run_dir / "config"

    for d in (models_dir, logs_dir, rollouts_dir, config_dir):
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
        "rollouts": rollouts_dir,
        "config": config_dir,
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


def _safe_float(x):
    try:
        return float(x)
    except TypeError:
        return None


def save_run_config(config_dir: Path, scenario: ScenarioConfig, model: PPO) -> None:
    """
    Save scenario configuration, PPO hyperparameters AND REWARDS as JSON.
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
        "rewards": {
            # Phase 1: L1 station-keeping
            "w_pos": W_POS,
            "w_vel": W_VEL,
            "w_ctrl": W_CTRL,
            "deadband": L1_DEADBAND,
            "far_limit": L1_FAR_LIMIT,
            "crash_penalty": CRASH_PENALTY,
            "planar_z_threshold": PLANAR_Z_THRESHOLD,
            "planar_vz_threshold": PLANAR_VZ_THRESHOLD,
            "w_planar": W_PLANAR,
            # Phase 2: reference-orbit tracking
            "w_pos_ref": W_POS_REF,
            "w_vel_ref": W_VEL_REF,
            "w_ctrl_ref": W_CTRL_REF,
            "w_planar_ref": W_PLANAR_REF,
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


# ======================================================================
# Rollouts as CSV
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
    """
    env = Cr3bpStationKeepingEnv(scenario=scenario, use_reference_orbit=use_reference_orbit)

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0

    records: list[dict[str, Any]] = []

    dt = env.dt
    dim = env.dim

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)

        rel_pos = obs[:dim]
        rel_vel = obs[dim : 2 * dim]

        pos = rel_pos + env.target

        if use_reference_orbit and hasattr(env, "vel_ref_current"):
            vel = rel_vel + env.vel_ref_current
        else:
            vel = rel_vel

        dv = info.get("dv", np.zeros(dim))

        t = step * dt

        row: dict[str, Any] = {"t": t, "reward": float(reward)}

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
    print(f"[INFO] Rollout written to: {csv_path}")

    env.close()


# ======================================================================
# Custom Callbacks
# ======================================================================


class TqdmProgressCallback(BaseCallback):
    """
    Progress bar callback using tqdm.
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class SimulationRecorderCallback(BaseCallback):
    """
    Callback that records full simulations at selected rollout indices.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        rollouts_dir: Path,
        verbose: int = 0,
        n_sim_first: int = 20,
        power_start: int = 3,
        power_end: int = 11,
        max_steps: int | None = None,
        use_reference_orbit: bool = False,
    ):
        super().__init__(verbose)
        self.scenario = scenario
        self.rollouts_dir = rollouts_dir
        self.n_sim_first = n_sim_first
        self.max_steps = max_steps
        self.use_reference_orbit = use_reference_orbit

        self.special_indices = {2**n for n in range(power_start, power_end + 1)}

        self.sim_index = 0

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("[SimulationRecorder] Training started, recording enabled.")

    def _on_rollout_end(self) -> None:
        self.sim_index += 1

        should_record = (
            self.sim_index <= self.n_sim_first or self.sim_index in self.special_indices
        )

        if not should_record:
            return

        csv_name = f"sim_{self.sim_index:03d}_steps_{self.num_timesteps}.csv"
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
# Environment factory
# ======================================================================


def make_env_fn(
    scenario: ScenarioConfig,
    seed_offset: int = 0,
    use_reference_orbit: bool = False,
):
    """
    Factory for environment instances compatible with :class:`DummyVecEnv`.
    """

    def _thunk():
        env = Cr3bpStationKeepingEnv(
            scenario=scenario,
            use_reference_orbit=use_reference_orbit,
            seed=seed_offset,
        )
        env = Monitor(env)
        return env

    return _thunk


# ======================================================================
# Training entry point
# ======================================================================


def train(
    scenario_name: str = "earth-moon-L1-3D",
    use_reference_orbit: bool = False,
    init_model_path: str | None = None,
    learning_rate: float = 3e-4,
) -> Path:
    """
    Train a PPO agent on the given scenario.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name!r}")

    scenario = SCENARIOS[scenario_name]
    print(
        f"[INFO] Starting training for scenario: {scenario.name} "
        f"(use_reference_orbit={use_reference_orbit})"
    )

    run_dirs = make_run_dirs(scenario)
    models_dir = run_dirs["models"]
    logs_dir = run_dirs["logs"]
    rollouts_dir = run_dirs["rollouts"]
    config_dir = run_dirs["config"]

    env = DummyVecEnv(
        [make_env_fn(scenario, i, use_reference_orbit) for i in range(N_ENVS)]
    )

    device = "cpu"
    print("Using device:", device)

    if init_model_path is not None:
        print(f"[INFO] Loading initial model from: {init_model_path}")
        model = PPO.load(
            init_model_path,
            env=env,
            device=device,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(logs_dir / "tb"),
            learning_rate=learning_rate,
            n_steps=2048 // max(N_ENVS, 1),
            batch_size=64,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // max(N_ENVS, 1),
        save_path=str(models_dir),
        name_prefix="checkpoint",
    )

    eval_env = Cr3bpStationKeepingEnv(
        scenario=scenario,
        use_reference_orbit=use_reference_orbit,
    )
    eval_env = Monitor(eval_env, filename=str(logs_dir / "eval_monitor.csv"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=20_000 // max(N_ENVS, 1),
        deterministic=True,
        render=False,
    )

    progress_callback = TqdmProgressCallback(total_timesteps=TOTAL_TIMESTEPS)

    sim_recorder = SimulationRecorderCallback(
        scenario=scenario,
        rollouts_dir=rollouts_dir,
        verbose=1,
        n_sim_first=20,
        power_start=3,
        power_end=6,
        max_steps=None,
        use_reference_orbit=use_reference_orbit,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, progress_callback, sim_recorder],
    )

    last_model_path = models_dir / "ppo_last.zip"
    model.save(str(last_model_path))
    print(f"[INFO] Last model saved to: {last_model_path}")

    save_run_config(config_dir, scenario, model)
    export_eval_to_csv(Path(logs_dir), logs_dir / "eval_rewards.csv")

    best_model_path = models_dir / "best_model.zip"
    chosen_ckpt: Path
    if best_model_path.exists():
        chosen_ckpt = best_model_path
        print(f"[INFO] Found best_model.zip at: {best_model_path}")
    else:
        chosen_ckpt = last_model_path
        print(
            f"[WARN] No best_model.zip found in {models_dir}, "
            f"using last model instead: {last_model_path}"
        )

    model_best = PPO.load(
        chosen_ckpt,
        env=Cr3bpStationKeepingEnv(
            scenario=scenario,
            use_reference_orbit=use_reference_orbit,
        ),
        device="cpu",
    )
    final_csv = rollouts_dir / "best_policy_final_rollout.csv"
    rollout_policy_to_csv(
        scenario=scenario,
        model=model_best,
        csv_path=final_csv,
        max_steps=None,
        deterministic=True,
        use_reference_orbit=use_reference_orbit,
    )

    env.close()
    eval_env.close()

    return chosen_ckpt


if __name__ == "__main__":
    print("Starting single-phase training (Halo orbit tracking only) with device=cpu")

    scenario_name = "earth-moon-L1-3D"

    train(
        scenario_name=scenario_name,
        use_reference_orbit=True,
        init_model_path=None,
        learning_rate=3e-4,
    )
