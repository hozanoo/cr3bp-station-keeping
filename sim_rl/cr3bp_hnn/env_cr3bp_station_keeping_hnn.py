"""
HNN-accelerated Gymnasium environment for CR3BP station-keeping.

This environment replaces the numerical integrator (DOP853/RK45) with
a trained Hamiltonian Neural Network (HNN) to predict the next state.
This allows for significantly faster training of the RL agent.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from sim_rl.cr3bp.constants import (
    SYSTEMS, LAGRANGE_POINTS, DT, MAX_STEPS,
    L1_DEADBAND, L1_FAR_LIMIT, CRASH_RADIUS_PRIMARY1, CRASH_RADIUS_PRIMARY2, CRASH_PENALTY,
    SCALE_POS, SCALE_VEL,
    PLANAR_Z_THRESHOLD, PLANAR_VZ_THRESHOLD,
    # HNN Specific Constants
    HNN_MODEL_FILENAME, HNN_META_FILENAME,
    W_POS_HNN, W_VEL_HNN, W_CTRL_HNN, W_PLANAR_HNN
)
from sim_rl.cr3bp.scenarios import ScenarioConfig
from sim_rl.cr3bp.reference_orbits.halo_generator import HaloOrbitConfig, generate_halo_orbit

# Import HNN Model Architecture
from hnn_models.model.hnn import HamiltonianNN


class Cr3bpStationKeepingEnvHNN(gym.Env):
    """
    CR3BP Environment powered by a Physics-Informed HNN.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig,
        dt: float = DT,
        max_steps: int = MAX_STEPS,
        max_dv: float = 0.005,
        seed: int | None = None,
        use_reference_orbit: bool = True,
    ) -> None:
        super().__init__()

        self.scenario = scenario
        self.system_id = scenario.system
        self.lagrange_point_id = scenario.lagrange_point
        self.dim = scenario.dim
        self.mu = SYSTEMS[self.system_id]["mu"]
        self.dt = dt
        self.max_steps = max_steps
        self.max_dv = max_dv
        self.use_reference_orbit = use_reference_orbit

        # --- HNN LOADING ---
        self.device = torch.device("cpu") # CPU is fine for inference
        self._load_hnn_model()

        # --- ORBIT SETUP ---
        target_3d = LAGRANGE_POINTS[self.system_id][self.lagrange_point_id]
        if self.dim == 2:
            self.lagrange_pos = target_3d[:2].astype(np.float64)
        else:
            self.lagrange_pos = target_3d.astype(np.float64)

        self.halo_ref = None
        self.halo_len = 0
        self._halo_index = 0
        self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        if self.use_reference_orbit:
            # Load or generate Halo Orbit
            # Path logic: sim_rl/cr3bp_hnn/env... -> ../cr3bp/reference_orbits
            halo_dir = Path(__file__).resolve().parents[1] / "cr3bp" / "reference_orbits" / "data"
            halo_dir.mkdir(parents=True, exist_ok=True)
            halo_filename = f"halo_{self.system_id}_{self.lagrange_point_id}.npy"
            halo_path = halo_dir / halo_filename
            
            if halo_path.exists():
                self.halo_ref = np.load(halo_path)
            else:
                cfg = HaloOrbitConfig(
                    system_id=self.system_id, 
                    lagrange_point=self.lagrange_point_id,
                    z_amplitude=0.08, 
                    steps_per_period=2000, 
                    dt=self.dt
                )
                self.halo_ref = generate_halo_orbit(cfg, save_path=halo_path)
            self.halo_len = self.halo_ref.shape[0]

        # Use HNN specific reward weights
        self.w_pos = float(W_POS_HNN)
        self.w_vel = float(W_VEL_HNN)
        self.w_ctrl = float(W_CTRL_HNN)
        self.w_planar = float(W_PLANAR_HNN)
        
        self.deadband = 0.0 if use_reference_orbit else float(L1_DEADBAND)
        self.far_limit = float(L1_FAR_LIMIT)

        # Target Setup
        if self.use_reference_orbit and self.halo_ref is not None:
            ref0 = self.halo_ref[0]
            self.target = ref0[: self.dim].astype(np.float64)
            self.vel_ref_current = ref0[self.dim : 2 * self.dim].astype(np.float64)
        else:
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        # Spaces
        self.action_dim = self.dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # Observation: [scaled_pos, scaled_vel]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.dim,), dtype=np.float32)

        # Internal State (Physical)
        self.state_phys = np.zeros(2 * self.dim, dtype=np.float64)
        self.step_count = 0
        
        self.np_random = None
        self.seed(seed)

    def _load_hnn_model(self):
        """Load HNN weights and normalization stats (Hybrid Load)."""
        # Path logic: sim_rl/cr3bp_hnn/env... -> ../../hnn_models
        project_root = Path(__file__).resolve().parents[2] 
        ckpt_dir = project_root / "hnn_models" / "checkpoints"
        
        meta_path = ckpt_dir / HNN_META_FILENAME
        ckpt_path = ckpt_dir / HNN_MODEL_FILENAME

        if not meta_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"HNN files missing in {ckpt_dir}. Please run HNN training first.")

        # 1. Load Meta (Normalization)
        with meta_path.open("r") as f:
            meta = json.load(f)
        
        # 2. Init Model
        self.hnn_model = HamiltonianNN(dim=self.dim, hidden_dims=meta["hidden_dims"], activation="sine")
        
        # 3. Set Normalization (from Meta)
        mean = torch.tensor(meta["state_mean"], dtype=torch.float32)
        std = torch.tensor(meta["state_std"], dtype=torch.float32)
        self.hnn_model.set_state_normalization(mean, std)
        
        # 4. Load Weights (from Checkpoint)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.hnn_model.load_state_dict(state_dict)
        
        # Re-enforce normalization (safety)
        self.hnn_model.set_state_normalization(mean, std)
        
        self.hnn_model.to(self.device)
        self.hnn_model.eval()

    def seed(self, seed: int | None = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Initial State: On Halo Orbit (random phase)
        if self.use_reference_orbit and self.halo_ref is not None:
            if self.np_random is not None:
                self._halo_index = int(self.np_random.integers(0, self.halo_len))
            else:
                self._halo_index = 0
            
            # Start exactly on reference
            start_state = self.halo_ref[self._halo_index].copy()
            
            # Update target
            self.target = start_state[:self.dim]
            self.vel_ref_current = start_state[self.dim:]
        else:
            start_state = np.concatenate([self.lagrange_pos, np.zeros(self.dim)])

        # Apply random noise (Robustness)
        if self.np_random is not None:
            start_state += self.np_random.normal(scale=self.scenario.pos_noise, size=2*self.dim)

        self.state_phys = start_state
        self.step_count = 0
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        # Calculate Relative State
        pos = self.state_phys[:self.dim]
        vel = self.state_phys[self.dim:]
        
        rel_pos = pos - self.target
        rel_vel = vel - self.vel_ref_current
        
        # Apply Hard Scaling (Deterministic)
        scaled_pos = rel_pos * SCALE_POS
        scaled_vel = rel_vel * SCALE_VEL
        
        return np.concatenate([scaled_pos, scaled_vel]).astype(np.float32)

    def step(self, action):
        # 1. Apply Action (Delta-V)
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        dv = np.zeros(self.dim)
        dv[:] = action * self.max_dv
        
        # Add dv to current velocity
        self.state_phys[self.dim:] += dv
        
        # 2. HNN Integration (RK4 Step)
        y_canonical = self._pv_to_canonical(self.state_phys)
        y_next_canonical = self._rk4_step_hnn(y_canonical, self.dt)
        self.state_phys = self._canonical_to_pv(y_next_canonical)
        
        self.step_count += 1
        
        # 3. Update Reference Target (Halo moves)
        if self.use_reference_orbit:
            self._halo_index = (self._halo_index + 1) % self.halo_len
            ref_next = self.halo_ref[self._halo_index]
            self.target = ref_next[:self.dim]
            self.vel_ref_current = ref_next[self.dim:]
            
        # 4. Calculate Rewards & Done
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(obs, dv)
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, info

    def _pv_to_canonical(self, pv):
        q, v = pv[:self.dim], pv[self.dim:]
        p = v.copy()
        p[0] = v[0] - q[1]; p[1] = v[1] + q[0] 
        return np.concatenate([q, p])

    def _canonical_to_pv(self, qp):
        q, p = qp[:self.dim], qp[self.dim:]
        v = p.copy()
        v[0] = p[0] + q[1]; v[1] = p[1] - q[0]
        return np.concatenate([q, v])

    def _rk4_step_hnn(self, y0, dt):
        y_t = torch.tensor(y0, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        def f(y):
            q, p = y[:, :self.dim], y[:, self.dim:]
            with torch.enable_grad():
                q.requires_grad_(True); p.requires_grad_(True)
                dq, dp = self.hnn_model.time_derivatives(q, p)
            return torch.cat([dq, dp], dim=-1)
        
        k1 = f(y_t)
        k2 = f(y_t + 0.5 * dt * k1)
        k3 = f(y_t + 0.5 * dt * k2)
        k4 = f(y_t + dt * k3)
        
        y_next = y_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return y_next.squeeze(0).detach().cpu().numpy()

    def _compute_reward(self, obs, dv):
        scaled_pos = obs[:self.dim]
        scaled_vel = obs[self.dim:]
        
        rel_pos = scaled_pos / SCALE_POS
        rel_vel = scaled_vel / SCALE_VEL
        
        dist = np.linalg.norm(rel_pos)
        vel_err = np.linalg.norm(rel_vel)
        ctrl_effort = np.linalg.norm(dv)
        
        cost = (self.w_pos * dist**2) + (self.w_vel * vel_err) + (self.w_ctrl * ctrl_effort)
        
        if self.dim == 3:
            pos = rel_pos + self.target
            vel = rel_vel + self.vel_ref_current
            if abs(pos[2]) < PLANAR_Z_THRESHOLD and abs(vel[2]) < PLANAR_VZ_THRESHOLD:
                cost += self.w_planar

        reward = -cost
        
        pos_abs = rel_pos + self.target
        r1 = np.linalg.norm(pos_abs - np.array([-self.mu, 0, 0]))
        r2 = np.linalg.norm(pos_abs - np.array([1-self.mu, 0, 0]))
        
        terminated = False
        if r1 < CRASH_RADIUS_PRIMARY1 or r2 < CRASH_RADIUS_PRIMARY2:
            terminated = True
            reward -= CRASH_PENALTY
        elif dist > self.far_limit:
            reward -= (self.w_pos * (dist - self.far_limit)**2)
            
        return reward, terminated, {"dist": dist, "dv": dv}