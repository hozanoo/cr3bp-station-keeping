"""
Gymnasium-compatible CR3BP station-keeping environment (HNN Robust Version).

This environment combines:
1. Physics-Informed Neural Network (HNN) for dynamics propagation (replacing numerical integrator).
2. Robustness features (Domain Randomization, Actuator Noise, Wind Disturbance).
3. Hybrid Reward Structure (Quadratic Position, Linear Control) for stability and sparsity.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from sim_rl.cr3bp.constants import (
    SYSTEMS,
    LAGRANGE_POINTS,
    BASE_START_STATES,
    DT,
    MAX_STEPS_HNN,
    L1_DEADBAND,
    L1_FAR_LIMIT,
    CRASH_RADIUS_PRIMARY1,
    CRASH_RADIUS_PRIMARY2,
    CRASH_PENALTY,
    PLANAR_Z_THRESHOLD,
    PLANAR_VZ_THRESHOLD,
    # Standard weights (fallback)
    W_POS, W_VEL, W_CTRL, W_PLANAR,
    # Repo specific constants (Scaling)
    SCALE_POS,
    SCALE_VEL,
    # Robust specific constants
    W_POS_ROBUST,
    W_VEL_ROBUST,
    W_CTRL_ROBUST,
    W_PLANAR_ROBUST,
    HALO_DEADBAND,
    HALO_DEADBAND_INNER_WEIGHT,
    # Noise params
    MU_UNCERTAINTY_PERCENT,
    ACTUATOR_NOISE_MAG,
    ACTUATOR_NOISE_ANGLE,
    DISTURBANCE_ACC_MAG,
    approximate_l1_x,
    approximate_l2_x,
    # HNN constants
    HNN_MODEL_FILENAME,
    HNN_META_FILENAME
)
from sim_rl.cr3bp.reference_orbits.halo_generator import HaloOrbitConfig, generate_halo_orbit
from sim_rl.cr3bp.scenarios import ScenarioConfig
from hnn_models.model.hnn import HamiltonianNN


class Cr3bpStationKeepingEnvHNNRobust(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig,
        dt: float = DT,
        max_steps: int = MAX_STEPS_HNN,
        max_dv: float = 0.005,
        integrator: str = "RK45",
        seed: int | None = None,
        use_reference_orbit: bool = True,
    ) -> None:
        super().__init__()

        self.scenario = scenario
        self.system_id = scenario.system
        self.lagrange_point_id = scenario.lagrange_point
        self.dim = scenario.dim
        self.action_mode = scenario.action_mode

        self.nominal_mu = SYSTEMS[self.system_id]["mu"]
        self.mu = self.nominal_mu

        self.dt = dt
        self.max_steps = max_steps
        self.max_dv = max_dv

        # Robustness configuration
        self.mu_uncertainty = MU_UNCERTAINTY_PERCENT
        self.act_noise_mag = ACTUATOR_NOISE_MAG
        self.act_noise_angle = ACTUATOR_NOISE_ANGLE
        self.dist_acc_mag = DISTURBANCE_ACC_MAG

        self.current_disturbance = np.zeros(3, dtype=float)

        # HNN Loading
        self.device = torch.device("cpu")
        self._load_hnn_model()

        target_3d = LAGRANGE_POINTS[self.system_id][self.lagrange_point_id]
        if self.dim == 2:
            self.lagrange_pos = target_3d[:2].astype(np.float64)
        else:
            self.lagrange_pos = target_3d.astype(np.float64)

        self.use_reference_orbit = use_reference_orbit
        self.halo_ref = None
        self.halo_len = 0
        self._halo_index = 0
        self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        if self.use_reference_orbit:
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
                    x_offset=-0.04,
                    periods=2.0,
                    steps_per_period=2000,
                    dt=self.dt,
                )
                self.halo_ref = generate_halo_orbit(cfg, save_path=halo_path)
            self.halo_len = self.halo_ref.shape[0]

        # Weights & Deadband (Robust Logic)
        if self.use_reference_orbit:
            self.w_pos = float(W_POS_ROBUST)
            self.w_vel = float(W_VEL_ROBUST)
            self.w_ctrl = float(W_CTRL_ROBUST)
            self.w_planar = float(W_PLANAR_ROBUST)
            self.deadband = float(HALO_DEADBAND)
        else:
            self.w_pos = float(W_POS)
            self.w_vel = float(W_VEL)
            self.w_ctrl = float(W_CTRL)
            self.w_planar = float(W_PLANAR)
            self.deadband = float(L1_DEADBAND)
        
        self.far_limit = float(L1_FAR_LIMIT)

        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            ref0 = self.halo_ref[0]
            self.target = ref0[: self.dim].astype(np.float64)
            self.vel_ref_current = ref0[self.dim : 2 * self.dim].astype(np.float64)
        else:
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        if self.action_mode == "planar":
            act_dim = 2
        elif self.action_mode == "full_3d":
            act_dim = self.dim
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        self.action_dim = act_dim
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        obs_dim = 2 * self.dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.state_phys = np.zeros(2 * self.dim, dtype=np.float64)
        self.step_count: int = 0
        self.np_random = None
        self.seed(seed)

    def _load_hnn_model(self):
        project_root = Path(__file__).resolve().parents[2] 
        ckpt_dir = project_root / "hnn_models" / "checkpoints"
        
        meta_path = ckpt_dir / HNN_META_FILENAME
        ckpt_path = ckpt_dir / HNN_MODEL_FILENAME

        if not meta_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"HNN files missing in {ckpt_dir}.")

        with meta_path.open("r") as f:
            meta = json.load(f)
        
        self.hnn_model = HamiltonianNN(dim=self.dim, hidden_dims=meta["hidden_dims"], activation="sine")
        
        mean = torch.tensor(meta["state_mean"], dtype=torch.float32)
        std = torch.tensor(meta["state_std"], dtype=torch.float32)
        self.hnn_model.set_state_normalization(mean, std)
        
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.hnn_model.load_state_dict(state_dict)
        
        self.hnn_model.set_state_normalization(mean, std)
        
        self.hnn_model.to(self.device)
        self.hnn_model.eval()

    def _init_state(self, ic_type: str | None = None) -> np.ndarray:
        key = (self.system_id, self.lagrange_point_id, self.dim)
        if key not in BASE_START_STATES:
            raise KeyError(f"No BASE_START_STATE defined for key {key!r}.")

        base = BASE_START_STATES[key].astype(np.float64)
        
        pos0 = base[:self.dim].copy()
        vel0 = base[self.dim:].copy()

        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            idx = int(self._halo_index)
            ref = self.halo_ref[idx]
            pos0 = ref[: self.dim].astype(np.float64)
            vel0 = ref[self.dim : 2 * self.dim].astype(np.float64)

        if self.np_random is not None:
            pos0 += self.np_random.normal(scale=self.scenario.pos_noise, size=self.dim)
            vel0 += self.np_random.normal(scale=self.scenario.vel_noise, size=self.dim)

        return np.concatenate([pos0, vel0])

    def _get_obs(self) -> np.ndarray:
        pos = self.state_phys[:self.dim]
        vel = self.state_phys[self.dim:]
        
        rel_pos = pos - self.target
        
        if self.use_reference_orbit and self.halo_ref is not None:
            rel_vel = vel - self.vel_ref_current
        else:
            rel_vel = vel

        scaled_pos = rel_pos * SCALE_POS
        scaled_vel = rel_vel * SCALE_VEL

        obs = np.concatenate([scaled_pos, scaled_vel]).astype(np.float32)
        return obs

    def seed(self, seed: int | None = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        ic_type = None
        if options is not None:
            ic_type = options.get("ic_type")

        if self.mu_uncertainty > 0.0 and self.np_random is not None:
            delta = self.np_random.uniform(-self.mu_uncertainty, self.mu_uncertainty)
            self.mu = self.nominal_mu * (1.0 + delta)
        else:
            self.mu = self.nominal_mu

        if self.dist_acc_mag > 0.0 and self.np_random is not None:
            u = self.np_random.normal(0, 1, size=3)
            norm = np.linalg.norm(u)
            if norm > 1e-6:
                u /= norm
            else:
                u = np.array([1.0, 0.0, 0.0])
            self.current_disturbance = u * self.dist_acc_mag
            if self.dim == 2:
                self.current_disturbance[2] = 0.0
        else:
            self.current_disturbance = np.zeros(3)

        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            if self.np_random is not None:
                self._halo_index = int(self.np_random.integers(0, self.halo_len))
            else:
                self._halo_index = 0
        else:
            self._halo_index = 0

        self.state_phys = self._init_state(ic_type=ic_type)
        self.step_count = 0

        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            ref = self.halo_ref[self._halo_index]
            self.target = ref[: self.dim].astype(np.float64)
            self.vel_ref_current = ref[self.dim : 2 * self.dim].astype(np.float64)
        else:
            if self.lagrange_point_id == "L1":
                lx = approximate_l1_x(self.mu)
            elif self.lagrange_point_id == "L2":
                lx = approximate_l2_x(self.mu)
            else:
                lx = self.lagrange_pos[0]
            self.lagrange_pos[0] = lx
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        obs = self._get_obs()
        info: dict = {
            "ic_type": ic_type or "l1_cloud",
            "perturbed_mu": self.mu,
            "disturbance_vec": self.current_disturbance
        }
        return obs, info

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

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        dv = np.zeros(self.dim, dtype=float)
        if self.action_mode == "planar":
            dv[:2] = action * self.max_dv
        elif self.action_mode == "full_3d":
            dv[:] = action * self.max_dv
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        # Robustness: Actuator Noise
        if self.np_random is not None:
            mag_scale = 1.0 + self.np_random.normal(0.0, self.act_noise_mag)
            dv *= mag_scale
            if self.act_noise_angle > 0.0:
                dv_norm = np.linalg.norm(dv)
                if dv_norm > 1e-9:
                    angle_noise = self.np_random.normal(0.0, self.act_noise_angle, size=self.dim)
                    dv += angle_noise * dv_norm

        self.state_phys[self.dim:] += dv

        # HNN Propagation (Coordinate transformation + RK4)
        y_canonical = self._pv_to_canonical(self.state_phys)
        y_next_canonical = self._rk4_step_hnn(y_canonical, self.dt)
        self.state_phys = self._canonical_to_pv(y_next_canonical)

        # Robustness: External Disturbance
        if self.dist_acc_mag > 0.0:
            dist_dv = self.current_disturbance[:self.dim] * self.dt
            self.state_phys[self.dim:] += dist_dv

        pos = self.state_phys[:self.dim]
        vel = self.state_phys[self.dim:]

        self.step_count += 1

        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            self._halo_index = (self._halo_index + 1) % self.halo_len
            ref = self.halo_ref[self._halo_index]
            self.target = ref[: self.dim].astype(np.float64)
            self.vel_ref_current = ref[self.dim : 2 * self.dim].astype(np.float64)
        else:
            vel_ref = np.zeros(self.dim, dtype=float)
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current = vel_ref.astype(np.float64)

        obs = self._get_obs()

        # Rewards (Robust Version)
        rel_pos = pos - self.target
        rel_vel = vel - self.vel_ref_current
        
        dist_target = float(np.linalg.norm(rel_pos))

        if self.dim == 2:
            primary1_pos = np.array([-self.mu, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0], dtype=float)
        else:
            primary1_pos = np.array([-self.mu, 0.0, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0, 0.0], dtype=float)

        dist_p1 = float(np.linalg.norm(pos - primary1_pos))
        dist_p2 = float(np.linalg.norm(pos - primary2_pos))

        if self.use_reference_orbit or self.deadband <= 0.0:
             pos_penalty = self.w_pos * (dist_target**2)
        else:
            if dist_target <= self.deadband:
                pos_penalty = 0.0
            else:
                excess = dist_target - self.deadband
                pos_penalty = self.w_pos * (excess**2)

        if self.use_reference_orbit and self.halo_ref is not None:
            vel_penalty = self.w_vel * float(np.linalg.norm(rel_vel))
        else:
            vel_penalty = 0.0

        ctrl_penalty = self.w_ctrl * float(np.linalg.norm(dv))

        far_penalty = 0.0
        if (not self.use_reference_orbit) and dist_target > self.far_limit:
            far_penalty = self.w_pos * (dist_target - self.far_limit) ** 2

        planar_penalty = 0.0
        if self.dim == 3:
            z_pos = pos[2]
            vz_val = vel[2]
            if abs(z_pos) < PLANAR_Z_THRESHOLD and abs(vz_val) < PLANAR_VZ_THRESHOLD:
                planar_penalty = self.w_planar

        reward = -(pos_penalty + vel_penalty + ctrl_penalty + far_penalty + planar_penalty)

        crash_p1 = dist_p1 < CRASH_RADIUS_PRIMARY1
        crash_p2 = dist_p2 < CRASH_RADIUS_PRIMARY2
        crashed = crash_p1 or crash_p2

        terminated = False
        truncated = False

        if crashed:
            terminated = True
            reward = -CRASH_PENALTY
        elif self.step_count >= self.max_steps:
            truncated = True

        info = {
            "dist_target": dist_target,
            "pos_penalty": pos_penalty,
            "vel_penalty": vel_penalty,
            "ctrl_penalty": ctrl_penalty,
            "dv": dv,
            "acc": np.zeros(3),
            "mu_actual": self.mu,
            "disturbance": self.current_disturbance
        }

        return obs, reward, terminated, truncated, info