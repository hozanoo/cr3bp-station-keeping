"""Gymnasium environment for station-keeping in the CR3BP.

This module defines :class:`Cr3bpStationKeepingEnv`, a Gymnasium-compatible
environment that simulates station-keeping around a selected Lagrange point
in the circular restricted three-body problem (CR3BP). The environment
supports both 2D (planar) and 3D dynamics in the rotating (synodic) frame.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .N_Body.nbody_lib import Body, NBodySystem, Simulator
from .constants import (
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
from .scenarios import ScenarioConfig


class Cr3bpStationKeepingEnv(gym.Env):
    """CR3BP station-keeping environment (2D/3D) for Gymnasium.

    The environment simulates a single massless spacecraft in the circular
    restricted three-body problem (CR3BP) using a rotating (synodic) frame
    with angular velocity :math:`\\omega = 1`. Two massive primaries
    (e.g. Earth–Moon or Earth–Sun) are fixed on the x-axis in the rotating
    frame, and the spacecraft is controlled via impulsive-like
    :math:`\\Delta v` actions.

    The configuration is provided via a :class:`ScenarioConfig` instance,
    which specifies:

    - the system (``"earth-moon"`` or ``"earth-sun"``),
    - the Lagrange point (``"L1"`` or ``"L2"``),
    - the dimensionality (2D or 3D),
    - the action mode (planar or full 3D),
    - the amount of position/velocity noise for domain randomization.

    Observations
    ------------
    The observation is a flat NumPy array of shape ``(2 * dim,)``:

    ``[dpos, dvel]``

    where

    - ``dpos`` is the position of the spacecraft relative to the chosen
      Lagrange point,
    - ``dvel`` is the velocity of the spacecraft in the rotating frame.

    Actions
    -------
    The action space is a continuous Box in ``[-1, 1]`` for each controlled
    component. Actions are interpreted as normalized :math:`\\Delta v`
    commands and are scaled by ``max_dv``.

    Reward
    ------
    The reward is a negative cost composed of:

    - position penalty outside a deadband around the Lagrange point,
    - velocity penalty,
    - control effort penalty (norm of :math:`\\Delta v`),
    - additional penalty when the spacecraft is far from the target.

    Episodes terminate when the spacecraft crashes into one of the primaries
    (distance below a system-specific crash radius) or when the maximum
    number of steps is reached.

    Parameters
    ----------
    scenario :
        Configuration describing the CR3BP system, Lagrange point and
        dimensionality.
    dt :
        Integration time step in normalized CR3BP units.
    max_steps :
        Maximum number of environment steps per episode.
    max_dv :
        Maximum :math:`\\Delta v` magnitude per step (in normalized units).
    seed :
        Optional random seed for Gymnasium's RNG.
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

        # Target Lagrange point position (2D or 3D)
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
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

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
    # Helpers
    # ------------------------------------------------------------------

    def _make_system(self) -> NBodySystem:
        """Create the underlying CR3BP N-body system.

        This constructs an :class:`NBodySystem` with two primaries and one
        massless spacecraft. The spacecraft is initialized in a
        halo-/Lyapunov-like orbit around the configured Lagrange point,
        using a predefined base state plus small Gaussian perturbations
        (domain randomization).

        Returns
        -------
        NBodySystem
            The initialized N-body system in the rotating frame.
        """
        mu = self.mu

        # Primary masses in normalized units
        primary_masses = [1.0 - mu, mu]

        # Primary positions on the x-axis in the rotating frame
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

        # Base initial state (position + velocity) for this scenario
        key = (self.system_id, self.lagrange_point_id, self.dim)
        if key not in BASE_START_STATES:
            raise KeyError(f"No BASE_START_STATE defined for key {key}.")

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
        """Compute the current observation.

        The observation is the spacecraft position and velocity in the
        rotating frame, expressed relative to the target Lagrange point.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2 * dim,)`` containing ``[dpos, dvel]``.
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
        """Set the random seed for domain randomization.

        Parameters
        ----------
        seed :
            Optional seed passed to Gymnasium's seeding utility.

        Returns
        -------
        list[int | None]
            A list containing the used seed, for Gymnasium compatibility.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment to a new initial state.

        This creates a fresh :class:`NBodySystem`, reinitializes the
        simulator and step counter, and returns the initial observation.

        Parameters
        ----------
        seed :
            Optional seed to reset the RNG for this episode.
        options :
            Additional environment-specific options (ignored).

        Returns
        -------
        tuple[numpy.ndarray, dict]
            Initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.system = self._make_system()
        self.sim = Simulator(self.system)
        self.step_count = 0
        obs = self._get_obs()
        info: dict = {}
        return obs, info

    def step(self, action):
        """Advance the environment by one time step.

        The step logic performs the following operations:

        1. Clip and scale the action in ``[-1, 1]`` to a physical
           :math:`\\Delta v` vector and apply it to the spacecraft velocity.
        2. Integrate the N-body system from ``t = 0`` to ``t = dt`` using
           an RK45 integrator.
        3. Compute the reward based on:
           - distance to the Lagrange point (with a deadband),
           - spacecraft velocity,
           - control effort (norm of :math:`\\Delta v`),
           - additional penalty when far from the target.
        4. Check termination conditions:
           - crash into a primary body (hard penalty, ``terminated = True``),
           - episode length reaching ``max_steps`` (``truncated = True``).

        Parameters
        ----------
        action :
            Normalized action array provided by the agent.

        Returns
        -------
        tuple[numpy.ndarray, float, bool, bool, dict]
            Observation, reward, terminated flag, truncated flag, and
            an info dictionary with diagnostic quantities.
        """
        # Clip action to valid range
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Build full-dimensional delta-v vector
        dv = np.zeros(self.dim, dtype=float)
        if self.action_mode == "planar":
            # Control only x and y
            dv[:2] = action * self.max_dv
        elif self.action_mode == "full_3d":
            dv[:] = action * self.max_dv
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        sat = self.system.bodies[0]
        sat.velocity = sat.velocity + dv

        # Integrate the N-body system over dt with RK45
        sol = self.sim.run(
            t_span=(0.0, self.dt),
            num_steps=2,
            solver_name="RK45",
        )
        final = sol.y[:, -1]

        # final contains [pos, vel] of all bodies flattened
        # Here we have only one body (the spacecraft)
        pos = final[: self.dim]
        vel = final[self.dim : 2 * self.dim]

        sat.position = pos
        sat.velocity = vel

        self.step_count += 1

        # New observation
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

        # ------------------ Reward components -------------------------

        # 1) Position penalty outside the deadband
        if dist_target <= L1_DEADBAND:
            pos_penalty = 0.0
        else:
            excess = dist_target - L1_DEADBAND
            pos_penalty = W_POS * (excess**2)

        # 2) Velocity penalty
        vel_penalty = W_VEL * float(np.linalg.norm(rel_vel))

        # 3) Control penalty (delta-v)
        ctrl_penalty = W_CTRL * float(np.linalg.norm(dv))

        # 4) Additional "far from target" penalty
        far_penalty = 0.0
        if dist_target > L1_FAR_LIMIT:
            far_penalty = W_POS * (dist_target - L1_FAR_LIMIT) ** 2

        reward = -(
            pos_penalty
            + vel_penalty
            + ctrl_penalty
            + far_penalty
        )

        # ------------------ Termination -------------------------------

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
            # important for later delta-v logging and analysis
            "dv": dv,
        }

        return obs, reward, terminated, truncated, info
