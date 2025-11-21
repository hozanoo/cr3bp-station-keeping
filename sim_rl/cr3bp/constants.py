"""Physical and configuration constants for the CR3BP environment.

This module contains normalized parameters for the circular restricted
three-body problem (CR3BP), including:
- Earth–Moon and Earth–Sun mass ratios,
- integration step sizes,
- Lagrange point approximations,
- default initial states for training scenarios,
- reward weights used in the station-keeping environment.

All quantities are expressed in normalized CR3BP units unless otherwise noted.
"""

import numpy as np

# =====================================================================
# CR3BP physical constants (normalized units)
# =====================================================================

#: Normalized mass ratio for the Earth–Moon CR3BP.
MU_EARTH_MOON: float = 0.012150585609624

#: Approximate normalized mass ratio for the Earth–Sun CR3BP.
MU_EARTH_SUN: float = 3.00348959632e-6

#: Default integration time step for the rotating-frame CR3BP.
DT: float = 0.01

#: Maximum number of simulation steps per episode.
MAX_STEPS: int = 700


# =====================================================================
# L1 / L2 approximations
# =====================================================================

def approximate_l1_x(mu: float) -> float:
    """Approximate the x-coordinate of L1 in the CR3BP.

    This simple analytic approximation is sufficient for reinforcement
    learning purposes, where precise values are not required.

    Parameters
    ----------
    mu : float
        Mass ratio of the CR3BP system.

    Returns
    -------
    float
        Approximate x-coordinate of the L1 point.
    """
    return 1.0 - (mu / 3.0) ** (1.0 / 3.0)


def approximate_l2_x(mu: float) -> float:
    """Approximate the x-coordinate of L2 in the CR3BP.

    Parameters
    ----------
    mu : float
        Mass ratio of the CR3BP system.

    Returns
    -------
    float
        Approximate x-coordinate of the L2 point.
    """
    return 1.0 + (mu / 3.0) ** (1.0 / 3.0)


# =====================================================================
# System configuration for CR3BP
# =====================================================================

#: Mass ratios for each supported CR3BP system.
SYSTEMS: dict[str, dict] = {
    "earth-moon": {"mu": MU_EARTH_MOON},
    "earth-sun": {"mu": MU_EARTH_SUN},
}

#: Precomputed Lagrange point coordinates for supported systems.
#: Points are stored in 3D (z = 0 for 2D scenarios).
LAGRANGE_POINTS: dict[str, dict[str, np.ndarray]] = {
    "earth-moon": {
        "L1": np.array([approximate_l1_x(MU_EARTH_MOON), 0.0, 0.0], dtype=float),
        "L2": np.array([approximate_l2_x(MU_EARTH_MOON), 0.0, 0.0], dtype=float),
    },
    "earth-sun": {
        "L1": np.array([approximate_l1_x(MU_EARTH_SUN), 0.0, 0.0], dtype=float),
        "L2": np.array([approximate_l2_x(MU_EARTH_SUN), 0.0, 0.0], dtype=float),
    },
}


# =====================================================================
# Default initial states (halo-/Lyapunov-like)
# =====================================================================

#: Base initial states for each supported scenario.
#: Keys are tuples of (system_id, lagrange_point, dimension).
#:
#: - dim = 2 → [x, y, vx, vy]
#: - dim = 3 → [x, y, z, vx, vy, vz]
BASE_START_STATES: dict[tuple[str, str, int], np.ndarray] = {
    # Earth–Moon L1, planar (2D)
    ("earth-moon", "L1", 2): np.array(
        [
            LAGRANGE_POINTS["earth-moon"]["L1"][0] - 0.01,  # x
            0.0,                                            # y
            0.0,                                            # vx
            0.20,                                           # vy
        ],
        dtype=float,
    ),

    # Earth–Moon L1, 3D (slight z-offset → halo-like)
    ("earth-moon", "L1", 3): np.array(
        [
            LAGRANGE_POINTS["earth-moon"]["L1"][0] - 0.04,  # x
            0.0,                                            # y
            0.08,                                           # z
            0.0,                                            # vx
            0.35,                                           # vy
            0.005,                                          # vz
        ],
        dtype=float,
    ),

    # Future extension example:
    # ("earth-sun", "L2", 3): ...
}


# =====================================================================
# Reward weights for the station-keeping environment
# =====================================================================

#: Weight for position error outside the deadband.
W_POS: float = 1.0

#: Weight for velocity magnitude.
W_VEL: float = 0.1

#: Weight for control effort (‖Δv‖).
W_CTRL: float = 0.01

#: Radius of the no-penalty region around the Lagrange point.
L1_DEADBAND: float = 0.15

#: Threshold after which a stronger "far away" penalty applies.
L1_FAR_LIMIT: float = 0.5


# =====================================================================
# Crash detection parameters
# =====================================================================

#: Crash radius for primary body 1 (e.g. Earth or Sun).
CRASH_RADIUS_PRIMARY1: float = 0.03

#: Crash radius for primary body 2 (e.g. Moon or Earth).
CRASH_RADIUS_PRIMARY2: float = 0.02

#: Penalty applied when the spacecraft collides with either primary.
CRASH_PENALTY: float = 200.0
