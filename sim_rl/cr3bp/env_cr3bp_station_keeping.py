"""
Gymnasium-compatible CR3BP station-keeping environment.

This environment simulates a spacecraft in the rotating frame of the
Circular Restricted Three Body Problem (CR3BP) near a selected
Lagrange point (for example Earth–Moon L1) and exposes it as a
continuous-control RL task.

The state consists of position and velocity relative to the chosen
Lagrange point, and the action is a delta-v command.
"""

from __future__ import annotations

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
    W_POS,
    W_VEL,
    W_CTRL,
    L1_DEADBAND,
    L1_FAR_LIMIT,
    CRASH_RADIUS_PRIMARY1,
    CRASH_RADIUS_PRIMARY2,
    CRASH_PENALTY,
)
from sim_rl.cr3bp.scenarios import ScenarioConfig


class Cr3bpStationKeepingEnv(gym.Env):
    """
    Station-keeping environment in the normalized CR3BP.

    The environment supports both 2D and 3D configurations and can be
    parameterized via :class:`ScenarioConfig`.

    Frame
    -----
    Rotating synodic frame with angular velocity :math:`\\omega = 1`.

    Observation
    -----------
    Concatenated vector ``[dpos, dvel]`` where:

    * ``dpos`` – position relative to the selected Lagrange point,
    * ``dvel`` – absolute velocity in the rotating frame.

    Action
    ------
    Continuous delta-v command. The action is scaled from ``[-1, 1]``
    to a physical delta-v using ``max_dv``.

    Reward
    ------
    The reward penalizes:

    * distance from the target (outside a deadband),
    * velocity magnitude,
    * control effort (delta-v norm),
    * excessive distance beyond ``L1_FAR_LIMIT``.

    Episodes terminate on:

    * crash into a primary body, or
    * reaching ``max_steps``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioConfig,
        dt: float = DT,
        max_steps: int = MAX_STEPS,
        max_dv: float = 0.005,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.scenario = scenario
        self.system_id = scenario.system
        self.lagrange_point_id = scenario.lagrange_point
        self.dim = scenario.dim
        self.action_mode = scenario.action_mode

        self.mu = SYSTEMS[self.system_id]["mu"]
        self.dt = dt
        self.max_steps = max_steps
        self.max_dv = max_dv

        # Target Lagrange point (2D or 3D)
        target_3d = LAGRANGE_POINTS[self.system_id][self.lagrange_point_id]
        if self.dim == 2:
            self.target = target_3d[:2].astype(np.float64)
        else:
            self.target = target_3d.astype(np.float64)

        # Action space
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
            shape=(act_dim,),
            dtype=np.float32,
        )

        # Observation space: [dpos, dvel]
        obs_dim = 2 * self.dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.system: NBodySystem | None = None
        self.sim: Simulator | None = None
        self.step_count: int = 0

        # Gymnasium RNG
        self.np_random = None
        self.seed(seed)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _make_system(self) -> NBodySystem:
        """
        Construct the underlying :class:`NBodySystem`.

        The system contains two primaries and one spacecraft. The
        spacecraft is initialized in a halo-/Lyapunov-like orbit near
        the chosen Lagrange point with small random perturbations added
        to position and velocity.
        """
        mu = self.mu

        # Primary masses (normalized)
        primary_masses = [1.0 - mu, mu]

        # Primary positions in the rotating frame (x-axis)
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

        # Baseline initial state
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

        # Domain randomization: small perturbations in all components
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

        system = NBodySystem(
            bodies=[sat],
            G=1.0,
            softening_factor=0.0,
            frame="rotating",
            omega=1.0,
            primary_masses=primary_masses,
            primary_positions=primary_positions,
        )
        return system

    def _get_obs(self) -> np.ndarray:
        """
        Compute the current observation vector.

        Returns
        -------
        numpy.ndarray
            Concatenated vector ``[dpos, dvel]`` as ``float32``.
        """
        sat = self.system.bodies[0]
        rel_pos = sat.position - self.target
        rel_vel = sat.velocity
        obs = np.concatenate([rel_pos, rel_vel]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None):
        """
        Initialize the environment's random number generator.

        Parameters
        ----------
        seed:
            Seed passed to Gymnasium's seeding utility.

        Returns
        -------
        list[int | None]
            List containing the seed.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to an initial state.

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Observation and info dictionary.
        """
        super().reset(seed=seed)
        self.system = self._make_system()
        self.sim = Simulator(self.system)
        self.step_count = 0
        obs = self._get_obs()
        info: dict = {}
        return obs, info

    def step(self, action):
        """
        Perform one integration step of length ``dt``.

        The action is interpreted as a normalized delta-v in the range
        ``[-1, 1]`` and scaled by ``max_dv``.

        Parameters
        ----------
        action:
            Action sampled by the agent or policy.

        Returns
        -------
        obs: numpy.ndarray
            Next observation.
        reward: float
            Scalar reward for this transition.
        terminated: bool
            ``True`` if a terminal state has been reached (for example crash).
        truncated: bool
            ``True`` if the episode was truncated by a time limit.
        info: dict
            Additional diagnostics and logging information.
        """
        # Clip and scale action
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        dv = np.zeros(self.dim, dtype=float)
        if self.action_mode == "planar":
            dv[:2] = action * self.max_dv
        elif self.action_mode == "full_3d":
            dv[:] = action * self.max_dv
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        sat = self.system.bodies[0]
        sat.velocity = sat.velocity + dv

        # Integrate over dt using RK45 (via Simulator)
        sol = self.sim.run(
            t_span=(0.0, self.dt),
            num_steps=2,
            solver_name="RK45",
        )
        final = sol.y[:, -1]

        # final contains [pos, vel] for all bodies (flattened).
        # With a single satellite body this is straightforward.
        pos = final[: self.dim]
        vel = final[self.dim : 2 * self.dim]

        # Acceleration from the CR3BP ODE at the final state
        rhs = self.system.differential_equation(sol.t[-1], final)
        acc = rhs[self.dim : 2 * self.dim]

        sat.position = pos
        sat.velocity = vel

        self.step_count += 1

        obs = self._get_obs()
        rel_pos = obs[: self.dim]
        rel_vel = obs[self.dim : 2 * self.dim]

        dist_target = float(np.linalg.norm(rel_pos))

        # Primary positions for crash detection
        if self.dim == 2:
            primary1_pos = np.array([-self.mu, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0], dtype=float)
        else:
            primary1_pos = np.array([-self.mu, 0.0, 0.0], dtype=float)
            primary2_pos = np.array([1.0 - self.mu, 0.0, 0.0], dtype=float)

        dist_p1 = float(np.linalg.norm(sat.position - primary1_pos))
        dist_p2 = float(np.linalg.norm(sat.position - primary2_pos))

        # ------------------ reward components ---------------------------

        # 1) Position penalty: outside the deadband
        if dist_target <= L1_DEADBAND:
            pos_penalty = 0.0
        else:
            excess = dist_target - L1_DEADBAND
            pos_penalty = W_POS * (excess**2)

        # 2) Velocity penalty
        vel_penalty = W_VEL * float(np.linalg.norm(rel_vel))

        # 3) Fuel penalty (delta-v)
        ctrl_penalty = W_CTRL * float(np.linalg.norm(dv))

        # 4) Extra penalty for large distance from target
        far_penalty = 0.0
        if dist_target > L1_FAR_LIMIT:
            far_penalty = W_POS * (dist_target - L1_FAR_LIMIT) ** 2

        reward = -(pos_penalty + vel_penalty + ctrl_penalty + far_penalty)

        # ------------------ termination logic ---------------------------

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
            "crash_primary1": crash_p1,
            "crash_primary2": crash_p2,
            "dv": dv,
            "acc": acc,
        }

        return obs, reward, terminated, truncated, info
