"""
Physical and reward-related constants for the CR3BP station-keeping environment.

This module collects all normalized CR3BP parameters, Lagrange point
approximations, baseline initial states and reward weights used by
:mod:`sim_rl.cr3bp.env_cr3bp_station_keeping`.
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# Physical constants in normalized CR3BP units
# =====================================================================

#: Mass ratio for the Earth–Moon CR3BP (mu = m2 / (m1 + m2)).
MU_EARTH_MOON: float = 0.012150585609624

#: Approximate mass ratio for the Earth–Sun CR3BP.
MU_EARTH_SUN: float = 3.00348959632e-6

#: Time step in the rotating, normalized CR3BP system.
DT: float = 0.01

#: Maximum number of environment steps per episode.
MAX_STEPS: int = 6000


# =====================================================================
# L1 / L2 approximations
# =====================================================================

def approximate_l1_x(mu: float) -> float:
    """
    Return a simple approximation for the x-position of L1 in the CR3BP.

    Parameters
    ----------
    mu:
        Mass ratio of the system.

    Returns
    -------
    float
        Approximate x-coordinate of L1 in normalized units.
    """
    return 1.0 - (mu / 3.0) ** (1.0 / 3.0)


def approximate_l2_x(mu: float) -> float:
    """
    Return a simple approximation for the x-position of L2 in the CR3BP.

    Parameters
    ----------
    mu:
        Mass ratio of the system.

    Returns
    -------
    float
        Approximate x-coordinate of L2 in normalized units.
    """
    return 1.0 + (mu / 3.0) ** (1.0 / 3.0)


# =====================================================================
# System parameters (Earth–Moon, Earth–Sun)
# =====================================================================

SYSTEMS: dict[str, dict[str, float]] = {
    "earth-moon": {
        "mu": MU_EARTH_MOON,
    },
    "earth-sun": {
        "mu": MU_EARTH_SUN,
    },
}

# Lagrange points as 3D coordinates (z = 0 as default)
LAGRANGE_POINTS: dict[str, dict[str, np.ndarray]] = {
    "earth-moon": {
        "L1": np.array(
            [approximate_l1_x(MU_EARTH_MOON), 0.0, 0.0],
            dtype=float,
        ),
        "L2": np.array(
            [approximate_l2_x(MU_EARTH_MOON), 0.0, 0.0],
            dtype=float,
        ),
    },
    "earth-sun": {
        "L1": np.array(
            [approximate_l1_x(MU_EARTH_SUN), 0.0, 0.0],
            dtype=float,
        ),
        "L2": np.array(
            [approximate_l2_x(MU_EARTH_SUN), 0.0, 0.0],
            dtype=float,
        ),
    },
}

# =====================================================================
# Baseline initial states (halo-/Lyapunov-like)
# =====================================================================

# Key: (system_id, lagrange_point, dim)
# Values:
#   dim = 2 → [x, y, vx, vy]
#   dim = 3 → [x, y, z, vx, vy, vz]

BASE_START_STATES: dict[tuple[str, str, int], np.ndarray] = {
    # Earth–Moon, L1, 2D
    ("earth-moon", "L1", 2): np.array(
        [
            LAGRANGE_POINTS["earth-moon"]["L1"][0] - 0.01,  # x
            0.0,                                            # y
            0.0,                                            # vx
            0.20,                                           # vy (approximately tangential)
        ],
        dtype=float,
    ),

    # Earth–Moon, L1, 3D (small z-offset → halo-like)
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

    # Placeholders for future extensions, e.g.:
    # ("earth-sun", "L2", 3): ...
}

# =====================================================================
# Reward weights
# =====================================================================

#: Position penalty outside the deadband region.
W_POS: float = 1.0

#: Velocity penalty.
W_VEL: float = 0.1

#: Control penalty (norm of delta-v).
W_CTRL: float = 0.01

#: Deadband radius around the L1 point.
L1_DEADBAND: float = 0.15

#: Soft limit for "far away" from the target.
L1_FAR_LIMIT: float = 0.5


# =====================================================================
# Crash logic (primaries)
# =====================================================================

#: Generic crash radius for primary body 1 in normalized CR3BP units.
CRASH_RADIUS_PRIMARY1: float = 0.03

#: Generic crash radius for primary body 2 in normalized CR3BP units.
CRASH_RADIUS_PRIMARY2: float = 0.02

#: Hard penalty applied on crash.
CRASH_PENALTY: float = 200.0
