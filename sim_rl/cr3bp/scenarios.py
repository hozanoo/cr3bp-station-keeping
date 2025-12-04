"""
Scenario configuration for CR3BP station-keeping experiments.

The :class:`ScenarioConfig` dataclass describes which physical system,
Lagrange point, dimensionality and action mode are used for a given setup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """
    Configuration for a single CR3BP station-keeping scenario.

    Parameters
    ----------
    name:
        Human-readable scenario name (for directory and logging).
    system:
        Physical system identifier (for example ``"earth-moon"``).
    lagrange_point:
        Lagrange point identifier (for example ``"L1"`` or ``"L2"``).
    dim:
        State dimensionality: 2 for planar, 3 for full 3D.
    action_mode:
        Action space type: ``"planar"`` or ``"full_3d"``.
    pos_noise:
        Standard deviation of initial position perturbation.
    vel_noise:
        Standard deviation of initial velocity perturbation.
    """
    name: str
    system: str
    lagrange_point: str
    dim: int
    action_mode: str
    pos_noise: float
    vel_noise: float


SCENARIOS: dict[str, ScenarioConfig] = {
    # Main 3D scenario: Earth–Moon system, L1, full 3D control.
    "earth-moon-L1-3D": ScenarioConfig(
        name="earth-moon-L1-3D",
        system="earth-moon",
        lagrange_point="L1",
        dim=3,
        action_mode="full_3d",
        pos_noise=0.0,
        vel_noise=0.0,
    ),

    # Phase-3 scenario: halo-reference dataset along the stored halo orbit.
    "earth-moon-L1-3D_halo_ref": ScenarioConfig(
        name="earth-moon-L1-3D_halo_ref",
        system="earth-moon",
        lagrange_point="L1",
        dim=3,
        action_mode="full_3d",
        pos_noise=0.0,
        vel_noise=0.0,
    ),

    # Example for a possible future 2D scenario:
    # "earth-moon-L1-2D": ScenarioConfig(
    #     name="earth-moon-L1-2D",
    #     system="earth-moon",
    #     lagrange_point="L1",
    #     dim=2,
    #     action_mode="planar",
    #     pos_noise=1e-3,
    #     vel_noise=1e-3,
    # ),
}
