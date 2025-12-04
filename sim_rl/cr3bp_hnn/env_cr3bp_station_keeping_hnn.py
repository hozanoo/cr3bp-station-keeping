"""
Gymnasium-compatible CR3BP station-keeping environment using a learned
Hamiltonian Neural Network (HNN) as dynamics model.

This is a model-based counterpart to
:class:`sim_rl.cr3bp.env_cr3bp_station_keeping.Cr3bpStationKeepingEnv`.

The task definition (observations, rewards, termination) is intentionally
kept as close as possible to the true-physics environment. The only
difference is the dynamics backend: we integrate the learned Hamiltonian
instead of the analytic CR3BP equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from hnn_models.model.hnn import HamiltonianNN

from sim_rl.cr3bp.constants import (
    SYSTEMS,
    LAGRANGE_POINTS,
    DT,
    MAX_STEPS,
    W_POS,
    W_VEL,
    W_CTRL,
    L1_DEADBAND,
    L1_FAR_LIMIT,
    CRASH_RADIUS_PRIMARY1,
    CRASH_RADIUS_PRIMARY2,
    CRASH_PENALTY,
    PLANAR_Z_THRESHOLD,
    PLANAR_VZ_THRESHOLD,
    W_PLANAR,
    W_POS_REF,
    W_VEL_REF,
    W_CTRL_REF,
    W_PLANAR_REF,
)
from sim_rl.cr3bp.scenarios import ScenarioConfig
from sim_rl.cr3bp.reference_orbits.halo_generator import HaloOrbitConfig, generate_halo_orbit


# ---------------------------------------------------------------------------
# Helper functions: canonical <-> (pos, vel), Jacobi, HNN integration
# ---------------------------------------------------------------------------


def pv_to_canonical(state_pv: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Convert (x, y, z, vx, vy, vz) -> (x, y, z, px, py, pz) for the rotating CR3BP.

    Convention (must match your HNN training!):

        p_x = v_x - y
        p_y = v_y + x
        p_z = v_z
    """
    q = state_pv[:dim].copy()
    v = state_pv[dim:].copy()

    x, y = q[0], q[1]
    vx, vy = v[0], v[1]

    p = np.zeros_like(v)
    p[0] = vx - y
    p[1] = vy + x
    if dim == 3:
        p[2] = v[2]

    return np.concatenate([q, p])


def canonical_to_pv(state_qp: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Convert (x, y, z, px, py, pz) -> (x, y, z, vx, vy, vz) with the inverse mapping:

        v_x = p_x + y
        v_y = p_y - x
        v_z = p_z.
    """
    q = state_qp[:dim].copy()
    p = state_qp[dim:].copy()

    x, y = q[0], q[1]
    px, py = p[0], p[1]

    v = np.zeros_like(q)
    v[0] = px + y
    v[1] = py - x
    if dim == 3:
        v[2] = p[2]

    return np.concatenate([q, v])


def jacobi_from_pv(pos: np.ndarray, vel: np.ndarray, mu: float) -> float:
    """
    Jacobi constant in the normalized rotating CR3BP:

        U = 0.5 * (x^2 + y^2) + (1-mu)/r1 + mu/r2
        C = 2 * U - ||v||^2
    """
    x, y, z = pos
    vx, vy, vz = vel

    mu1 = 1.0 - mu
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - mu1) ** 2 + y**2 + z**2)

    U = 0.5 * (x**2 + y**2) + mu1 / r1 + mu / r2
    v2 = vx**2 + vy**2 + vz**2
    return 2.0 * U - v2


@torch.no_grad()
def hnn_time_derivatives(
    model: HamiltonianNN,
    q: torch.Tensor,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate dq/dt and dp/dt from the HNN for given q, p.

    This assumes `model.time_derivatives(q, p)` exists and returns (dqdt, dpdt).
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if p.dim() == 1:
        p = p.unsqueeze(0)

    # We temporarily enable gradients inside, since the HNN uses autograd internally.
    with torch.enable_grad():
        q_req = q.clone().detach().requires_grad_(True)
        p_req = p.clone().detach().requires_grad_(True)
        dq_dt, dp_dt = model.time_derivatives(q_req, p_req)

    return dq_dt.detach(), dp_dt.detach()


def rk4_step_hnn(
    model: HamiltonianNN,
    y_qp: torch.Tensor,
    dt: float,
    dim: int,
) -> torch.Tensor:
    """
    One explicit RK4 step in canonical coordinates (q, p).
    """

    if y_qp.dim() == 1:
        y_qp = y_qp.unsqueeze(0)

    def f(y_local: torch.Tensor) -> torch.Tensor:
        q_local = y_local[:, :dim]
        p_local = y_local[:, dim:]
        dq_dt, dp_dt = hnn_time_derivatives(model, q_local, p_local)
        return torch.cat([dq_dt, dp_dt], dim=-1)

    k1 = f(y_qp)
    k2 = f(y_qp + 0.5 * dt * k1)
    k3 = f(y_qp + 0.5 * dt * k2)
    k4 = f(y_qp + dt * k3)

    y_new = y_qp + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y_new.squeeze(0)


# ---------------------------------------------------------------------------
# HNN-based CR3BP station-keeping environment
# ---------------------------------------------------------------------------


class Cr3bpStationKeepingHnnEnv(gym.Env):
    """
    Station-keeping environment in the normalized CR3BP using a
    Hamiltonian Neural Network (HNN) as dynamics model.

    The observation, action and reward structure mirrors
    :class:`Cr3bpStationKeepingEnv` as closely as possible.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig,
        hnn_model: HamiltonianNN,
        *,
        dt: float = DT,
        max_steps: int = MAX_STEPS,
        max_dv: float = 0.005,
        seed: int | None = None,
        use_reference_orbit: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        # --- Scenario / basic config ---
        self.scenario = scenario
        self.system_id = scenario.system
        self.lagrange_point_id = scenario.lagrange_point
        self.dim = scenario.dim
        self.action_mode = scenario.action_mode

        self.mu = SYSTEMS[self.system_id]["mu"]
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.max_dv = float(max_dv)

        self.use_reference_orbit = use_reference_orbit

        # --- HNN model ---
        self.hnn_model = hnn_model.to(device)
        self.hnn_model.eval()
        self.device = torch.device(device)

        # --- Target / reference orbit handling ---
        target_3d = LAGRANGE_POINTS[self.system_id][self.lagrange_point_id]
        if self.dim == 2:
            self.lagrange_pos = target_3d[:2].astype(np.float64)
        else:
            self.lagrange_pos = target_3d.astype(np.float64)

        self.halo_ref: Optional[np.ndarray] = None
        self.halo_len: int = 0
        self._halo_index: int = 0
        self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        if self.use_reference_orbit:
            # reuse same file convention as true-physics env
            halo_filename = f"halo_{self.system_id}_{self.lagrange_point_id}.npy"
            # NOTE: adjust path if needed to match your layout
            halo_path = (
                # this mirrors sim_rl/cr3bp/reference_orbits/data
                # but we keep it simple here
                # you can import the same helper used in the original env
                __file__
            )  # TODO: replace by proper Path logic
            # Placeholder: user should load halo_ref externally or mirror original env

        # reward weights (reuse from constants)
        if self.use_reference_orbit:
            self.w_pos = float(W_POS_REF)
            self.w_vel = float(W_VEL_REF)
            self.w_ctrl = float(W_CTRL_REF)
            self.w_planar = float(W_PLANAR_REF)
        else:
            self.w_pos = float(W_POS)
            self.w_vel = float(W_VEL)
            self.w_ctrl = float(W_CTRL)
            self.w_planar = float(W_PLANAR)

        # deadband / far-limit as in true env
        if self.use_reference_orbit:
            self.deadband = 0.0
        else:
            self.deadband = float(L1_DEADBAND)
        self.far_limit = float(L1_FAR_LIMIT)

        # current target (L1 or reference orbit sample)
        self.target = self.lagrange_pos.copy()
        self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        # action space
        if self.action_mode == "planar":
            act_dim = 2
        elif self.action_mode == "full_3d":
            act_dim = self.dim
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        self.action_dim = act_dim
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # observation = [rel_pos, rel_vel]
        obs_dim = 2 * self.dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # internal state: pos, vel in rotating frame
        self.pos = np.zeros(self.dim, dtype=np.float64)
        self.vel = np.zeros(self.dim, dtype=np.float64)
        self.step_count: int = 0

        # RNG
        self.np_random = None
        self.seed(seed)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        rel_pos = self.pos - self.target
        if self.use_reference_orbit:
            rel_vel = self.vel - self.vel_ref_current
        else:
            rel_vel = self.vel
        obs = np.concatenate([rel_pos, rel_vel]).astype(np.float32)
        return obs

    def seed(self, seed: int | None = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Reset & step
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # For now: sample initial state similarly to true env.
        # You can refactor this to reuse the same BASE_START_STATES logic
        # or even read from DB / halo generator.

        ic_type = None
        if options is not None:
            ic_type = options.get("ic_type")

        # TODO: replace this by proper initial condition logic.
        # For now we just start exactly at L1 with small random noise:
        base_pos = self.lagrange_pos.copy()
        base_vel = np.zeros(self.dim, dtype=np.float64)

        if self.np_random is not None:
            base_pos += self.np_random.normal(
                scale=self.scenario.pos_noise,
                size=self.dim,
            )
            base_vel += self.np_random.normal(
                scale=self.scenario.vel_noise,
                size=self.dim,
            )

        self.pos = base_pos
        self.vel = base_vel
        self.step_count = 0

        # reference orbit target / vel_ref_current
        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            # start at random point on halo
            if self.np_random is not None:
                self._halo_index = int(self.np_random.integers(0, self.halo_len))
            else:
                self._halo_index = 0
            ref = self.halo_ref[self._halo_index]
            self.target = ref[: self.dim].astype(np.float64)
            self.vel_ref_current = ref[self.dim : 2 * self.dim].astype(np.float64)
        else:
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current[:] = 0.0

        obs = self._get_obs()
        info: dict = {"ic_type": ic_type or "l1_cloud"}
        return obs, info

    def step(self, action):
        # scale action to delta-v
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        dv = np.zeros(self.dim, dtype=np.float64)
        if self.action_mode == "planar":
            dv[:2] = action * self.max_dv
        elif self.action_mode == "full_3d":
            dv[:] = action * self.max_dv
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        # apply thruster kick in velocity
        self.vel = self.vel + dv

        # integrate HNN dynamics for dt in canonical coordinates
        state_pv = np.concatenate([self.pos, self.vel])
        state_qp = pv_to_canonical(state_pv, dim=self.dim)

        y_t = torch.from_numpy(state_qp.astype(np.float32)).to(self.device)
        y_next = rk4_step_hnn(self.hnn_model, y_t, self.dt, dim=self.dim)
        state_qp_next = y_next.detach().cpu().numpy()

        # back to pos/vel
        state_pv_next = canonical_to_pv(state_qp_next, dim=self.dim)
        self.pos = state_pv_next[: self.dim]
        self.vel = state_pv_next[self.dim : 2 * self.dim]

        self.step_count += 1

        # reference orbit update (very simple: step index + 1)
        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            self._halo_index = (self._halo_index + 1) % self.halo_len
            ref = self.halo_ref[self._halo_index]
            self.target = ref[: self.dim].astype(np.float64)
            self.vel_ref_current = ref[self.dim : 2 * self.dim].astype(np.float64)
        else:
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current[:] = 0.0

        # build observation
        obs = self._get_obs()
        rel_pos = obs[: self.dim]
        rel_vel = obs[self.dim : 2 * self.dim]

        dist_target = float(np.linalg.norm(rel_pos))

        # distances to primaries (for crash detection)
        if self.dim == 2:
            primary1_pos = np.array([-self.mu, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0], dtype=float)
        else:
            primary1_pos = np.array([-self.mu, 0.0, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0, 0.0], dtype=float)

        dist_p1 = float(np.linalg.norm(self.pos - primary1_pos))
        dist_p2 = float(np.linalg.norm(self.pos - primary2_pos))

        # --- reward terms (mirror true env) ---
        if self.use_reference_orbit or self.deadband <= 0.0:
            pos_penalty = self.w_pos * (dist_target**2)
        else:
            if dist_target <= self.deadband:
                pos_penalty = 0.0
            else:
                excess = dist_target - self.deadband
                pos_penalty = self.w_pos * (excess**2)

        if self.use_reference_orbit:
            vel_penalty = self.w_vel * float(np.linalg.norm(rel_vel))
        else:
            vel_penalty = 0.0

        ctrl_penalty = self.w_ctrl * float(np.linalg.norm(dv))

        far_penalty = 0.0
        if (not self.use_reference_orbit) and dist_target > self.far_limit:
            far_penalty = self.w_pos * (dist_target - self.far_limit) ** 2

        planar_penalty = 0.0
        if self.dim == 3:
            z_pos = self.pos[2]
            vz_val = self.vel[2]
            if abs(z_pos) < PLANAR_Z_THRESHOLD and abs(vz_val) < PLANAR_VZ_THRESHOLD:
                planar_penalty = self.w_planar

        reward = -(pos_penalty + vel_penalty + ctrl_penalty + far_penalty + planar_penalty)

        # crash & termination
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

        # --- diagnostic: HNN energy & Jacobi, trust region ---
        state_pv_for_diag = np.concatenate([self.pos, self.vel])
        C_hnn = jacobi_from_pv(self.pos, self.vel, self.mu)

        with torch.no_grad():
            y_qp_diag = torch.from_numpy(state_qp_next.astype(np.float32)).to(self.device)
            q_diag = y_qp_diag[: self.dim].unsqueeze(0)
            p_diag = y_qp_diag[self.dim : 2 * self.dim].unsqueeze(0)
            H_val = self.hnn_model.forward(q_diag, p_diag).item()

        # simple trust region: outside far_limit => violated
        hnn_trust_region_violated = dist_target > self.far_limit

        info = {
            "dist_target": dist_target,
            "dist_primary1": dist_p1,
            "dist_primary2": dist_p2,
            "pos_penalty": pos_penalty,
            "vel_penalty": vel_penalty,
            "ctrl_penalty": ctrl_penalty,
            "far_penalty": far_penalty,
            "planar_penalty": planar_penalty,
            "crash_primary1": crash_p1,
            "crash_primary2": crash_p2,
            "dv": dv,
            # HNN diagnostics
            "hnn_jacobi": C_hnn,
            "hnn_energy": H_val,
            "hnn_trust_region_violated": hnn_trust_region_violated,
        }

        return obs, reward, terminated, truncated, info
