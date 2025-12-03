"""
Generate a (quasi) periodic Halo orbit around L1 in the CR3BP
using a simple two-parameter shooting method.

This script:
- uses the CR3BP equations in the rotating frame,
- searches for symmetric Halo-like initial conditions via a 2-parameter
  shooting on (x0, vy0),
- constrains the half-period y=0 crossing to be near the L1 x-position,
- estimates the period from that crossing,
- integrates several periods and optionally saves the trajectory as .npy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Imports from your project
# ----------------------------------------------------------------------
try:
    from sim_rl.cr3bp.constants import SYSTEMS, LAGRANGE_POINTS
except ImportError:
    # Fallback if the script is run standalone from a different location
    import sys

    sys.path.append(str(Path(__file__).parents[3]))
    from sim_rl.cr3bp.constants import SYSTEMS, LAGRANGE_POINTS


LagrangePointName = Literal["L1", "L2"]


# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------
@dataclass
class HaloOrbitConfig:
    system_id: str
    lagrange_point: LagrangePointName = "L1"

    # geometric properties
    z_amplitude: float = 0.08  # Az

    # integration / sampling
    periods: float = 2.0
    dt: float = 0.01           # sampling step (very important for RL!)
    rtol: float = 1e-9
    atol: float = 1e-9

    # max time window for the shooting run (should cover at least half a period)
    t_max_shooting: float = 10.0


# ----------------------------------------------------------------------
# Helpers to read CR3BP parameters
# ----------------------------------------------------------------------
def _get_mu(system_id: str) -> float:
    sys_data = SYSTEMS[system_id]
    if isinstance(sys_data, dict):
        return float(sys_data["mu"])
    return float(sys_data.mu)


def _get_lagrange_x(system_id: str, lp: str) -> float:
    """Return x-coordinate of the chosen Lagrange point."""
    # Nested dictionary structure: LAGRANGE_POINTS["earth-moon"]["L1"]
    if system_id in LAGRANGE_POINTS and lp in LAGRANGE_POINTS[system_id]:
        pt = LAGRANGE_POINTS[system_id][lp]
        return float(pt[0])

    # Fallback if a flat key like "earth-moon_L1" is ever used
    key = f"{system_id}_{lp}"
    if key in LAGRANGE_POINTS:
        return float(LAGRANGE_POINTS[key][0])

    raise KeyError(f"Lagrange Point {lp} not found for system {system_id!r}")


# ----------------------------------------------------------------------
# CR3BP dynamics (rotating frame, normalized units)
# ----------------------------------------------------------------------
def cr3bp_equations(t: float, state: np.ndarray, mu: float) -> list[float]:
    """Right-hand side of the CR3BP ODEs in the rotating frame."""
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

    mu1 = 1.0 - mu

    ax = 2.0 * vy + x - mu1 * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = -2.0 * vx + y - mu1 * y / r1**3 - mu * y / r2**3
    az = -mu1 * z / r1**3 - mu * z / r2**3

    return [vx, vy, vz, ax, ay, az]


# ----------------------------------------------------------------------
# Event: y = 0 crossings (NOT terminal)
# ----------------------------------------------------------------------
def event_y_crossing(t: float, y: np.ndarray, mu: float) -> float:
    """Event function for y = 0 plane crossing."""
    return y[1]


# Detect crossings but do not stop integration
event_y_crossing.terminal = False
event_y_crossing.direction = 0  # both directions allowed


# ----------------------------------------------------------------------
# Differential correction: find symmetric L1-centered Halo IC
# ----------------------------------------------------------------------
def find_halo_initial_state(
    mu: float,
    x_L: float,
    Az: float,
    t_max: float,
) -> tuple[np.ndarray, float]:
    """
    Find symmetric Halo initial conditions (x0, 0, Az, 0, vy0, 0) via a
    simple 2-parameter shooting on (x0, vy0).

    The objective:
    - at the first crossing of y = 0 AFTER t > 0, we want
      vx ≈ 0 and vz ≈ 0 (perpendicular crossing),
    - and the x-position at that crossing should be near the L1 x-position.
    """

    # Initial guess: small offset from L1 and moderate tangential velocity
    guess_x0 = x_L - 0.04
    guess_vy0 = 0.20

    def objective(params: np.ndarray) -> float:
        x0_curr, vy0_curr = params
        initial_state = np.array([x0_curr, 0.0, Az, 0.0, vy0_curr, 0.0], dtype=float)

        sol = solve_ivp(
            fun=cr3bp_equations,
            t_span=(0.0, t_max),
            y0=initial_state,
            args=(mu,),
            events=event_y_crossing,
            rtol=1e-9,
            atol=1e-9,
        )

        crossings_t = sol.t_events[0]
        crossings_y = sol.y_events[0]

        # We want the first crossing AFTER t > 0 (ignore t=0)
        valid_indices = [i for i, t_cross in enumerate(crossings_t) if t_cross > 1e-3]
        if not valid_indices:
            # No valid crossing -> extremely bad candidate
            return float("inf")

        idx = valid_indices[0]
        final_state = crossings_y[idx]

        x_f = final_state[0]
        vx_f = final_state[3]
        vz_f = final_state[5]

        # 1) Perpendicular crossing (vx ≈ 0, vz ≈ 0)
        v_term = np.sqrt(vx_f**2 + vz_f**2)

        # 2) x at the crossing should be near L1
        x_term = (x_f - x_L) ** 2

        # 3) keep x0 reasonably close to the initial guess (stability for Nelder–Mead)
        center_term = (x0_curr - guess_x0) ** 2

        # Weights; can be tuned if needed
        w_x_cross = 5.0
        w_center = 1.0

        return float(v_term + w_x_cross * x_term + w_center * center_term)

    print(
        f"Searching Halo IC for Az = {Az:.4f} "
        f"(initial guess: x0={guess_x0:.4f}, vy0={guess_vy0:.4f})..."
    )

    res = minimize(
        objective,
        x0=np.array([guess_x0, guess_vy0], dtype=float),
        method="Nelder-Mead",
        tol=1e-6,
    )

    if not res.success:
        print(f"WARNING: optimizer did not fully converge: {res.message}")

    x_opt, vy_opt = res.x
    state0 = np.array([x_opt, 0.0, Az, 0.0, vy_opt, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # Estimate the period from the first y = 0 crossing after t > 0
    # ------------------------------------------------------------------
    final_run = solve_ivp(
        fun=cr3bp_equations,
        t_span=(0.0, t_max),
        y0=state0,
        args=(mu,),
        events=event_y_crossing,
        rtol=1e-9,
        atol=1e-9,
    )

    crossings_t = final_run.t_events[0]
    valid_indices = [i for i, t_cross in enumerate(crossings_t) if t_cross > 1e-3]

    if not valid_indices:
        raise RuntimeError(
            "Could not detect a half-period y=0 crossing for the optimized initial state. "
            "Try increasing t_max_shooting or adjusting the initial guess."
        )

    half_period = float(crossings_t[valid_indices[0]])
    period = 2.0 * half_period

    if period <= 0.0:
        raise RuntimeError(f"Estimated period is non-positive: {period:.6e}")

    return state0, period


# ----------------------------------------------------------------------
# Generate full Halo trajectory
# ----------------------------------------------------------------------
def generate_halo_orbit(
    config: HaloOrbitConfig,
    save_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Generate a (quasi) periodic Halo orbit for the given configuration.

    Parameters
    ----------
    config:
        HaloOrbitConfig with system, Lagrange point, z-amplitude, etc.
    save_path:
        Optional path where the resulting trajectory is saved as .npy.

    Returns
    -------
    traj : np.ndarray of shape (N, 6)
        State trajectory [x, y, z, vx, vy, vz] in the rotating frame,
        sampled with step config.dt.
    """
    mu = _get_mu(config.system_id)
    x_L = _get_lagrange_x(config.system_id, config.lagrange_point)

    state0, period = find_halo_initial_state(
        mu=mu,
        x_L=x_L,
        Az=config.z_amplitude,
        t_max=config.t_max_shooting,
    )

    print(
        f"-> Halo candidate found: T = {period:.6f}, "
        f"x0 = {state0[0]:.6f}, vy0 = {state0[4]:.6f}"
    )

    # Integrate over the requested number of periods
    t_end = config.periods * period

    # Very important for RL: sample with constant dt matching the env
    dt = float(config.dt)
    if dt <= 0.0:
        raise ValueError(f"config.dt must be positive, got {dt}")

    # ensure we include t_end (within numerical noise)
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)

    sol = solve_ivp(
        fun=cr3bp_equations,
        t_span=(0.0, t_end),
        y0=state0,
        args=(mu,),
        t_eval=t_eval,
        method="DOP853",
        rtol=config.rtol,
        atol=config.atol,
    )

    traj = sol.y.T  # shape (N, 6)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, traj)
        print(f"Saved Halo trajectory to {save_path} (shape={traj.shape})")

    return traj


# ----------------------------------------------------------------------
# Simple test run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cfg = HaloOrbitConfig(
        system_id="earth-moon",
        lagrange_point="L1",
        z_amplitude=0.08,
        periods=2.0,
        dt=0.01,  # should match your env DT
    )

    out_path = Path(__file__).parent / "data" / "halo_earth-moon_L1.npy"

    try:
        traj = generate_halo_orbit(cfg, save_path=out_path)
        print("Success!")
        print(f"Trajectory shape: {traj.shape}")
    except Exception as exc:
        print(f"Error during Halo generation: {exc}")
