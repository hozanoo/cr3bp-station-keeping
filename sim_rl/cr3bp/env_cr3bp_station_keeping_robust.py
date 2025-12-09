"""
Gymnasium-compatible CR3BP station-keeping environment (Robust Version).

This environment mirrors the reward structure of the deterministic 'Repo' version
(Quadratic Position, Linear Velocity, Linear Control) to ensure efficient
station-keeping behavior (sparsity), while adding physical robustness challenges:
1. Domain Randomization (perturbation of mu).
2. Actuator Noise (thrust magnitude and direction errors).
3. External Disturbances (simulated unmodeled constant forces like SRP).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim_rl.cr3bp.N_Body.nbody_lib import Body, NBodySystem, Simulator
from sim_rl.cr3bp.constants import (
    SYSTEMS,
    LAGRANGE_POINTS,
    BASE_START_STATES,
    DT,
    MAX_STEPS,
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
    # Robust specific constants (Rewards & Noise)
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
)
from sim_rl.cr3bp.reference_orbits.halo_generator import HaloOrbitConfig, generate_halo_orbit
from sim_rl.cr3bp.scenarios import ScenarioConfig


class Cr3bpStationKeepingEnvRobust(gym.Env):
    """
    Station-keeping environment in the normalized CR3BP (Robust Version).

    Features
    --------
    - Deterministic observation scaling (same as Repo).
    - Randomized physical parameters (mu) per episode.
    - Noisy actuators (magnitude and direction).
    - External disturbance forces (constant bias per episode).
    - Hybrid Reward Structure: Quadratic Position, Linear Velocity/Control.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig,
        dt: float = DT,
        max_steps: int = MAX_STEPS,
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

        # Nominal mu (base value)
        self.nominal_mu = SYSTEMS[self.system_id]["mu"]
        # Current mu (will be randomized in reset)
        self.mu = self.nominal_mu

        self.dt = dt
        self.max_steps = max_steps
        self.max_dv = max_dv
        self.integrator = integrator

        # Robustness configuration
        self.mu_uncertainty = MU_UNCERTAINTY_PERCENT
        self.act_noise_mag = ACTUATOR_NOISE_MAG
        self.act_noise_angle = ACTUATOR_NOISE_ANGLE
        self.dist_acc_mag = DISTURBANCE_ACC_MAG

        # Disturbance vector for current episode
        self.current_disturbance = np.zeros(3, dtype=float)

        # Standard L1/L2 locations (Nominal)
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

        # Load reference orbit if requested
        if self.use_reference_orbit:
            halo_dir = Path(__file__).resolve().parent / "reference_orbits" / "data"
            halo_dir.mkdir(parents=True, exist_ok=True)
            halo_filename = f"halo_{self.system_id}_{self.lagrange_point_id}.npy"
            halo_path = halo_dir / halo_filename
            
            if halo_path.exists():
                self.halo_ref = np.load(halo_path)
            else:
                # Generate if not exists
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

        # ------------------------------------------------------------------
        # Weights & Deadband
        # ------------------------------------------------------------------
        if self.use_reference_orbit:
            # Use specific ROBUST weights
            self.w_pos = float(W_POS_ROBUST)
            self.w_vel = float(W_VEL_ROBUST)
            self.w_ctrl = float(W_CTRL_ROBUST)
            self.w_planar = float(W_PLANAR_ROBUST)
            # Use Halo Deadband from constants (set to 0.0 in constants.py for strictness)
            self.deadband = float(HALO_DEADBAND)
        else:
            # Fallback to standard
            self.w_pos = float(W_POS)
            self.w_vel = float(W_VEL)
            self.w_ctrl = float(W_CTRL)
            self.w_planar = float(W_PLANAR)
            self.deadband = float(L1_DEADBAND)
        
        self.far_limit = float(L1_FAR_LIMIT)

        # Initialize Target
        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            ref0 = self.halo_ref[0]
            self.target = ref0[: self.dim].astype(np.float64)
            self.vel_ref_current = ref0[self.dim : 2 * self.dim].astype(np.float64)
        else:
            self.target = self.lagrange_pos.copy()
            self.vel_ref_current = np.zeros(self.dim, dtype=np.float64)

        # Spaces
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

        self.system: NBodySystem | None = None
        self.sim: Simulator | None = None
        self.step_count: int = 0
        self.np_random = None
        self.seed(seed)

    def _make_system(self, ic_type: str | None = None) -> NBodySystem:
        """
        Creates the NBodySystem using the current (potentially perturbed) mu.
        """
        mu = self.mu

        primary_masses = [1.0 - mu, mu]
        if self.dim == 2:
            primary_positions = [
                np.array([-mu, 0.0], dtype=float),
                np.array([1.0 - mu, 0.0], dtype=float),
            ]
        else:
            primary_positions = [
                np.array([-mu, 0.0, 0.0], dtype=float),
                np.array([1.0 - mu, 0.0, 0.0], dtype=float),
            ]

        # Initial State Generation
        key = (self.system_id, self.lagrange_point_id, self.dim)
        if key not in BASE_START_STATES:
            raise KeyError(f"No BASE_START_STATE defined for key {key!r}.")

        base = BASE_START_STATES[key].astype(np.float64)

        if self.dim == 2:
            pos0 = base[:2].copy()
            vel0 = base[2:4].copy()
        else:
            pos0 = base[:3].copy()
            vel0 = base[3:6].copy()

        # If we are using a reference orbit, snap to it
        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            idx = int(self._halo_index)
            ref = self.halo_ref[idx]
            pos0 = ref[: self.dim].astype(np.float64)
            vel0 = ref[self.dim : 2 * self.dim].astype(np.float64)

        # Add initial state noise
        if self.np_random is not None:
            pos0 += self.np_random.normal(
                scale=self.scenario.pos_noise,
                size=self.dim,
            )
            vel0 += self.np_random.normal(
                scale=self.scenario.vel_noise,
                size=self.dim,
            )

        sat = Body(
            mass=1.0,
            position=pos0,
            velocity=vel0,
            name="sat",
        )

        bodies = [sat]

        system = NBodySystem(
            bodies=bodies,
            G=1.0,
            softening_factor=0.0,
            frame="rotating",
            omega=1.0,
            primary_masses=primary_masses,
            primary_positions=primary_positions,
        )
        return system

    def _get_obs(self) -> np.ndarray:
        sat = self.system.bodies[0]
        rel_pos = sat.position - self.target

        if self.use_reference_orbit and self.halo_ref is not None:
            rel_vel = sat.velocity - self.vel_ref_current
        else:
            rel_vel = sat.velocity

        # Deterministic Scaling
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

        # ------------------------------------------------------------------
        # ROBUSTNESS: Domain Randomization (Mu Perturbation)
        # ------------------------------------------------------------------
        if self.mu_uncertainty > 0.0 and self.np_random is not None:
            delta = self.np_random.uniform(-self.mu_uncertainty, self.mu_uncertainty)
            self.mu = self.nominal_mu * (1.0 + delta)
        else:
            self.mu = self.nominal_mu

        # ------------------------------------------------------------------
        # ROBUSTNESS: Disturbance Forces (Constant per episode)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Halo Index Logic
        # ------------------------------------------------------------------
        if self.use_reference_orbit and self.halo_ref is not None and self.halo_len > 0:
            if self.np_random is not None:
                self._halo_index = int(self.np_random.integers(0, self.halo_len))
            else:
                self._halo_index = 0
        else:
            self._halo_index = 0

        # Create system with perturbed mu
        self.system = self._make_system(ic_type=ic_type)
        self.sim = Simulator(self.system)
        self.step_count = 0

        # Update target logic based on reference
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

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Calculate nominal DV
        dv = np.zeros(self.dim, dtype=float)
        if self.action_mode == "planar":
            dv[:2] = action * self.max_dv
        elif self.action_mode == "full_3d":
            dv[:] = action * self.max_dv
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        # ------------------------------------------------------------------
        # ROBUSTNESS: Actuator Noise (Thrust Errors)
        # ------------------------------------------------------------------
        if self.np_random is not None:
            # Magnitude noise
            mag_scale = 1.0 + self.np_random.normal(0.0, self.act_noise_mag)
            dv *= mag_scale

            # Direction noise
            if self.act_noise_angle > 0.0:
                dv_norm = np.linalg.norm(dv)
                if dv_norm > 1e-9:
                    angle_noise = self.np_random.normal(0.0, self.act_noise_angle, size=self.dim)
                    dv += angle_noise * dv_norm

        sat = self.system.bodies[0]
        sat.velocity = sat.velocity + dv

        # Run Integrator (Natural Dynamics)
        sol = self.sim.run(
            t_span=(0.0, self.dt),
            num_steps=2,
            solver_name=self.integrator,
        )
        final = sol.y[:, -1]

        # ------------------------------------------------------------------
        # ROBUSTNESS: External Disturbance (Constant Bias Impulse)
        # ------------------------------------------------------------------
        if self.dist_acc_mag > 0.0:
            dist_dv = self.current_disturbance[:self.dim] * self.dt
            final[self.dim : 2 * self.dim] += dist_dv

        pos = final[: self.dim]
        vel = final[self.dim : 2 * self.dim]

        rhs = self.system.differential_equation(sol.t[-1], final)
        acc = rhs[self.dim : 2 * self.dim]

        sat.position = pos
        sat.velocity = vel

        self.step_count += 1

        # Update Reference Orbit Target
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

        # ------------------------------------------------------------------
        # REWARDS: Hybrid structure matching Repo Version
        # Position: Quadratic (**2) for stability
        # Velocity: Linear (Norm) for damping
        # Control:  Linear (Norm) for sparsity/fuel saving
        # ------------------------------------------------------------------
        rel_pos = sat.position - self.target
        rel_vel = sat.velocity - self.vel_ref_current
        
        dist_target = float(np.linalg.norm(rel_pos))

        if self.dim == 2:
            primary1_pos = np.array([-self.mu, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0], dtype=float)
        else:
            primary1_pos = np.array([-self.mu, 0.0, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0, 0.0], dtype=float)

        dist_p1 = float(np.linalg.norm(sat.position - primary1_pos))
        dist_p2 = float(np.linalg.norm(sat.position - primary2_pos))

        # 1. Position Penalty (Quadratic)
        if self.use_reference_orbit:
            if dist_target <= self.deadband:
                # Inside deadband: Scaled quadratic penalty
                pos_penalty = self.w_pos * (dist_target**2) * HALO_DEADBAND_INNER_WEIGHT
            else:
                # Outside deadband: Quadratic penalty on excess
                excess = dist_target - self.deadband
                pos_penalty = self.w_pos * (excess**2)
        else:
            # Standard L1 logic
            if self.deadband <= 0.0:
                pos_penalty = self.w_pos * (dist_target**2)
            else:
                if dist_target <= self.deadband:
                    pos_penalty = 0.0
                else:
                    excess = dist_target - self.deadband
                    pos_penalty = self.w_pos * (excess**2)

        # 2. Velocity Penalty (Linear)
        if self.use_reference_orbit and self.halo_ref is not None:
            vel_penalty = self.w_vel * float(np.linalg.norm(rel_vel))
        else:
            vel_penalty = 0.0

        # 3. Control Penalty (Linear)
        # Using linear norm enforces sparsity (engines off behavior)
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
            "acc": acc,
            # Robust info logging
            "mu_actual": self.mu,
            "disturbance": self.current_disturbance
        }

        return obs, reward, terminated, truncated, info