"""Scenario configuration definitions for CR3BP station-keeping.

This module defines :class:`ScenarioConfig`, a dataclass that encapsulates all
parameters required to configure a station-keeping scenario in the CR3BP
environment. It also contains a registry of predefined scenarios that can be
used directly by the training pipeline or extended with new configurations.
"""

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """Configuration container for a CR3BP scenario.

    A scenario defines the physical system (Earth–Moon or Earth–Sun),
    the Lagrange point to target, the dimensionality of the simulation,
    the action mode, and the magnitude of the domain randomization noise.

    Parameters
    ----------
    name : str
        Unique scenario identifier (e.g. ``"earth-moon-L1-3D"``).
    system : str
        Name of the CR3BP system. Supported values are:
        - ``"earth-moon"``
        - ``"earth-sun"``
    lagrange_point : str
        Target Lagrange point. Supported values are:
        - ``"L1"``
        - ``"L2"``
    dim : int
        Dimensionality of the simulation (``2`` or ``3``).
    action_mode : str
        Determines which components of Δv can be controlled:
        - ``"planar"`` → control only x/y
        - ``"full_3d"`` → control all spatial dimensions
    pos_noise : float
        Standard deviation of the initial position perturbation for domain
        randomization.
    vel_noise : float
        Standard deviation of the initial velocity perturbation for domain
        randomization.
    """

    name: str
    system: str
    lagrange_point: str
    dim: int
    action_mode: str
    pos_noise: float
    vel_noise: float


#: Registry of available CR3BP scenarios.
#: Keys are scenario names, values are :class:`ScenarioConfig` instances.
SCENARIOS: dict[str, ScenarioConfig] = {
    # Main default scenario:
    # Earth–Moon CR3BP, L1 point, full 3D dynamics, 3D thrust,
    # and moderate domain randomization.
    "earth-moon-L1-3D": ScenarioConfig(
        name="earth-moon-L1-3D",
        system="earth-moon",
        lagrange_point="L1",
        dim=3,
        action_mode="full_3d",
        pos_noise=1e-3,
        vel_noise=1e-3,
    ),

    # Example for future extensions:
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
