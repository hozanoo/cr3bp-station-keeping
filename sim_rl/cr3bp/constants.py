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
MAX_STEPS: int = 1200


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
#   dim = 2 -> [x, y, vx, vy]
#   dim = 3 -> [x, y, z, vx, vy, vz]

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

    # Earth–Moon, L1, 3D (small z-offset -> halo-like)
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
}

# =====================================================================
# Reward weights (Legacy / Standard)
# =====================================================================

#: Position penalty outside the deadband region.
W_POS: float = 1.0

#: Velocity penalty.
W_VEL: float = 0.1

#: Control penalty (norm of delta-v).
W_CTRL: float = 0.5

#: Deadband radius around the L1 point.
L1_DEADBAND: float = 0.0

#: Soft limit for "far away" from the target.
L1_FAR_LIMIT: float = 0.25


# =====================================================================
# Crash logic (primaries)
# =====================================================================

#: Generic crash radius for primary body 1 in normalized CR3BP units.
CRASH_RADIUS_PRIMARY1: float = 0.03

#: Generic crash radius for primary body 2 in normalized CR3BP units.
CRASH_RADIUS_PRIMARY2: float = 0.02

#: Hard penalty applied on crash.
CRASH_PENALTY: float = 500.0


# =====================================================================
# Halo-Enforcing (Anti-Planar) Constants
# =====================================================================

#: Threshold for Z-position to be considered "in-plane".
PLANAR_Z_THRESHOLD: float = 0.02

#: Threshold for Z-velocity to be considered "stationary vertically".
PLANAR_VZ_THRESHOLD: float = 0.05

#: Penalty applied when both planar conditions are met.
W_PLANAR: float = 5.0

# =====================================================================
# Alternate reward weights for reference-orbit tracking (Phase 2)
# =====================================================================

#: Position penalty when tracking a reference Halo orbit.
W_POS_REF: float = 20.0

#: Velocity penalty when tracking a reference orbit.
W_VEL_REF: float = 0.05

#: Control penalty when tracking a reference orbit (discourage hovering).
W_CTRL_REF: float = 0.08

#: Planar penalty in reference-orbit mode.
W_PLANAR_REF: float = 0.0


# =====================================================================
# REPO VERSION: Deterministic Scaling & Rewards (Classic Integrator)
# =====================================================================

#: Hard scaling factor for Position input to the neural network.
SCALE_POS: float = 20.0

#: Hard scaling factor for Velocity input to the neural network.
SCALE_VEL: float = 20.0

#: Position penalty for Repo version.
W_POS_REPO: float = 20.0

#: Velocity penalty for Repo version.
W_VEL_REPO: float = 0.05

#: Control penalty for Repo version.
W_CTRL_REPO: float = 0.08

#: Planar penalty for Repo version.
W_PLANAR_REPO: float = 0.0


# =====================================================================
# HNN INTEGRATION & REWARDS (Domain C: Physics-Informed)
# =====================================================================

# Model filenames (Must exist in hnn_models/checkpoints/)
HNN_MODEL_FILENAME: str = "hnn_cr3bp_l1_halo_finetune_v3.pt"
HNN_META_FILENAME: str  = "hnn_cr3bp_l1_mixed_v3_meta.json"

# Separate reward weights for HNN-based training
W_POS_HNN: float = 20.0
W_VEL_HNN: float = 0.05
W_CTRL_HNN: float = 0.08
W_PLANAR_HNN: float = 0.0

# =====================================================================
# ROBUST VERSION: Stochastic & Disturbance Constants
# =====================================================================

#: Percentage of uncertainty applied to the mass parameter mu (e.g. 0.005 = 0.5%).
MU_UNCERTAINTY_PERCENT: float = 0.005

#: Standard deviation for actuator magnitude noise (percentage, e.g. 0.01 = 1%).
ACTUATOR_NOISE_MAG: float = 0.01

#: Standard deviation for actuator direction noise (radians).
ACTUATOR_NOISE_ANGLE: float = 0.005

#: Magnitude of the random constant disturbance acceleration (approximates SRP/unknown forces).
# Set to 0.1 for stress testing as requested. Normal range ~1e-4.
DISTURBANCE_ACC_MAG: float = 1.0e-4

#: Position penalty for Robust version.
W_POS_ROBUST: float = 20.0

#: Velocity penalty for Robust version.
W_VEL_ROBUST: float = 0.05

#: Control penalty for Robust version.
W_CTRL_ROBUST: float = 0.08

#: Planar penalty for Robust version.
W_PLANAR_ROBUST: float = 0.0

# =====================================================================
# ROBUST/REPO COMPATIBILITY (Reward Consistency)
# =====================================================================

#: Radius of the "tube" around the Halo orbit where errors are tolerated/weighted less.
HALO_DEADBAND: float = 0.005

#: Scaling factor for reward INSIDE the deadband (e.g. 0.001 = weak penalty).
HALO_DEADBAND_INNER_WEIGHT: float = 1.0e-3