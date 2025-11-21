"""Training script for PPO-based station-keeping in the CR3BP.

This module wires together the CR3BP environment, Stable-Baselines3 PPO,
and a small experiment management system that:

- creates a run directory structure under ``training/runs/``,
- stores checkpoints and TensorBoard logs,
- records evaluation statistics and exports them to CSV,
- periodically records full rollouts for later visualization.

The main entry point is :func:`train`, which can be called directly or
via ``python -m sim_rl.training.train_poc`` from the project root.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
    BaseCallback,
)

from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv
from sim_rl.cr3bp.scenarios import SCENARIOS, ScenarioConfig

# ----------------------------------------------------------------------
# Hyperparameters and global configuration
# ----------------------------------------------------------------------

#: Total number of PPO training timesteps.
TOTAL_TIMESTEPS = 8_000_000

#: Number of parallel environments for PPO.
N_ENVS = 4  # can be set to 1 if needed

#: Base directory for all runs, relative to this file (training/runs).
BASE_RUN_DIR = Path(__file__).parent / "runs"


# ======================================================================
# Run management utilities
# ======================================================================


def make_run_dirs(scenario: ScenarioConfig) -> dict[str, Path]:
    """Create the directory structure for a new training run.

    The structure is:

    ``runs/<scenario.name>/run_YYYYMMDD_HHMMSS/{models,logs,rollouts,config}``

    Parameters
    ----------
    scenario : ScenarioConfig
        Scenario for which the run is created.

    Returns
    -------
    dict[str, pathlib.Path]
        Dictionary with keys ``"run_dir"``, ``"models"``, ``"logs"``,
        ``"rollouts"``, and ``"config"`` pointing to the respective paths.
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

    # Optional pointer to the latest run
    latest_path = scenario_root / "latest_run.txt"
    try:
        latest_path.write_text(str(run_dir), encoding="utf-8")
    except Exception:
        # Not critical if this fails (e.g. permissions), so we ignore it.
        pass

    return {
        "run_dir": run_dir,
        "models": models_dir,
        "logs": logs_dir,
        "rollouts": rollouts_dir,
        "config": config_dir,
    }


def export_eval_to_csv(eval_log_dir: Path, csv_path: Path) -> None:
    """Export evaluation results from ``evaluations.npz`` to CSV.

    This expects the file produced by :class:`EvalCallback` and writes
    a CSV file containing timesteps, mean episode reward, and (if
    available) mean episode length.

    Parameters
    ----------
    eval_log_dir : pathlib.Path
        Directory containing ``evaluations.npz``.
    csv_path : pathlib.Path
        Output path for the CSV file.
    """
    npz_path = eval_log_dir / "evaluations.npz"
    if not npz_path.exists():
        print(f"[WARN] No evaluations.npz found in {eval_log_dir}.")
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
        Trained PPO model (used here only to read hyperparameters).
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


# ======================================================================
# Rollout export
# ======================================================================


def rollout_policy_to_csv(
    scenario: ScenarioConfig,
    model: PPO,
    csv_path: Path,
    max_steps: int | None = None,
    deterministic: bool = True,
) -> None:
    """Run a single rollout and store state and Δv history as CSV.

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
    # No Monitor wrapper: we want direct access to environment attributes
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

        # Absolute position in the rotating frame
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
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Rollout saved to: {csv_path}")

    env.close()


# ======================================================================
# Callback: periodic simulation recording
# ======================================================================


class SimulationRecorderCallback(BaseCallback):
    """Record full rollouts during training for later visualization.

    The callback records:

    - the first ``n_sim_first`` rollouts, and
    - additional rollouts whose indices are in ``{3**n | n in [power_start, power_end]}``.

    Each recorded rollout is saved as a CSV file in the ``rollouts`` folder
    of the current run directory.

    Parameters
    ----------
    scenario : ScenarioConfig
        Scenario for which the rollouts are recorded.
    rollouts_dir : pathlib.Path
        Directory where rollout CSV files are written.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = info).
    n_sim_first : int, optional
        Number of initial consecutive rollouts to record.
    power_start : int, optional
        Lower exponent for the special indices (base 3).
    power_end : int, optional
        Upper exponent for the special indices (base 3).
    max_steps : int or None, optional
        Optional cap on the number of steps per recorded rollout.
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
    ) -> None:
        super().__init__(verbose)
        self.scenario = scenario
        self.rollouts_dir = rollouts_dir
        self.n_sim_first = n_sim_first
        self.max_steps = max_steps

        # Indices 3^n (e.g. 27, 81, 243, 729, ...)
        self.special_indices = {3**n for n in range(power_start, power_end + 1)}

        # Counts how many simulations have been recorded so far
        self.sim_index = 0

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(
                "[SimulationRecorder] Training started, "
                "recording of selected rollouts is enabled."
            )

    def _on_rollout_end(self) -> None:
        """Hook called by SB3 after each PPO rollout.

        We decide here whether to record a new simulation based on
        ``n_sim_first`` and ``special_indices``.
        """
        self.sim_index += 1

        should_record = (
            self.sim_index <= self.n_sim_first
            or self.sim_index in self.special_indices
        )

        if not should_record:
            return

        # File name: sim_<index>_steps_<total_timesteps>.csv
        csv_name = f"sim_{self.sim_index:03d}_steps_{self.num_timesteps}.csv"
        csv_path = self.rollouts_dir / csv_name

        if self.verbose > 0:
            print(
                f"[SimulationRecorder] Recording simulation {self.sim_index} "
                f"at timestep={self.num_timesteps} → {csv_path}"
            )

        rollout_policy_to_csv(
            scenario=self.scenario,
            model=self.model,
            csv_path=csv_path,
            max_steps=self.max_steps,
            deterministic=True,
        )

    def _on_step(self) -> bool:
        """Mandatory hook for :class:`BaseCallback`.

        We do not need to perform any per-step logic here, so this always
        returns ``True`` to continue training.
        """
        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(
                "[SimulationRecorder] Training finished. "
                f"Recorded {self.sim_index} simulations in total."
            )


# ======================================================================
# Environment factory
# ======================================================================


def make_env_fn(scenario: ScenarioConfig, seed_offset: int = 0):
    """Factory function for CR3BP environments used in a vectorized setup.

    This returns a thunk compatible with :class:`DummyVecEnv`.

    Parameters
    ----------
    scenario : ScenarioConfig
        Scenario configuration for the environment.
    seed_offset : int, optional
        Optional offset for the environment seed (currently unused).

    Returns
    -------
    callable
        A function that, when called, creates a new monitored environment.
    """

    def _thunk():
        env = Cr3bpStationKeepingEnv(scenario=scenario)
        env = Monitor(env)
        return env

    return _thunk


# ======================================================================
# Training entry point
# ======================================================================


def train(scenario_name: str = "earth-moon-L1-3D") -> None:
    """Train a PPO policy for a given CR3BP scenario.

    Parameters
    ----------
    scenario_name : str, optional
        Key of the scenario to use from :data:`sim_rl.cr3bp.scenarios.SCENARIOS`.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]
    print(f"[INFO] Starting training for scenario: {scenario.name}")

    run_dirs = make_run_dirs(scenario)
    models_dir = run_dirs["models"]
    logs_dir = run_dirs["logs"]
    rollouts_dir = run_dirs["rollouts"]
    config_dir = run_dirs["config"]

    # Vectorized environment
    env = DummyVecEnv([make_env_fn(scenario, i) for i in range(N_ENVS)])

    device = "cpu"
    print("[INFO] Using device:", device)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(logs_dir / "tb"),
        learning_rate=3e-4,
        n_steps=2048 // max(N_ENVS, 1),  # per environment
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // max(N_ENVS, 1),
        save_path=str(models_dir),
        name_prefix="checkpoint",
    )

    # Evaluation environment (single)
    eval_env = Cr3bpStationKeepingEnv(scenario=scenario)
    eval_env = Monitor(eval_env, filename=str(logs_dir / "eval_monitor.csv"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=20_000 // max(N_ENVS, 1),
        deterministic=True,
        render=False,
    )

    progress_callback = ProgressBarCallback()

    # SimulationRecorder for presentation rollouts
    sim_recorder = SimulationRecorderCallback(
        scenario=scenario,
        rollouts_dir=rollouts_dir,
        verbose=1,
        n_sim_first=20,  # first N rollouts
        power_start=3,  # 3^3 = 27, 3^4 = 81, ...
        power_end=6,  # 27, 81, 243, 729
        max_steps=None,  # can be set to e.g. 600 to cap rollout length
    )

    # Training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, progress_callback, sim_recorder],
    )

    # Save last model
    last_model_path = models_dir / "ppo_last.zip"
    model.save(str(last_model_path))
    print(f"[INFO] Last model saved to: {last_model_path}")

    # Save run configuration
    save_run_config(config_dir, scenario, model)

    # Export evaluation results as CSV
    export_eval_to_csv(Path(logs_dir), logs_dir / "eval_rewards.csv")

    # Rollout with best model (if available)
    best_model_path = models_dir / "best_model.zip"
    if best_model_path.exists():
        final_csv = rollouts_dir / "best_policy_final_rollout.csv"
        rollout_policy_to_csv(
            scenario=scenario,
            model=PPO.load(
                best_model_path,
                env=Cr3bpStationKeepingEnv(scenario),
            ),
            csv_path=final_csv,
            max_steps=None,
            deterministic=True,
        )
    else:
        print(f"[WARN] No best_model.zip found in {models_dir}.")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    print("[INFO] Starting training with device=cpu (forced, PPO + MLP).")
    train("earth-moon-L1-3D")
