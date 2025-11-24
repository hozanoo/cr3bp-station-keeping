# hnn_models/utilities/integration.py
"""
Integration helpers for Hamiltonian Neural Networks.

Provides:
- hnn_time_derivatives: safe wrapper around HamiltonianNN.time_derivatives
  that avoids autograd graph blow-up during rollouts.
- leapfrog_step: symplectic integrator step for (q, p).
- integrate_leapfrog: multi-step integration loop for HNN dynamics.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from hnn_models.model.hnn import HamiltonianNN


def hnn_time_derivatives(
    model: HamiltonianNN,
    q: Tensor,
    p: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute dq/dt and dp/dt from the HNN in a rollout-safe way.

    The tensors are detached from any previous graph and re-marked
    as requiring gradients so that autograd can compute derivatives
    for the current step only (no graph growth across time).
    """
    # Detach from previous steps and re-enable grad tracking
    q = q.detach().requires_grad_(True)
    p = p.detach().requires_grad_(True)

    dq_dt, dp_dt = model.time_derivatives(q, p)
    return dq_dt, dp_dt


def leapfrog_step(
    model: HamiltonianNN,
    q: Tensor,
    p: Tensor,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a single symplectic Leapfrog step for the HNN.

    Parameters
    ----------
    model:
        Trained HamiltonianNN instance.
    q, p:
        Tensors of shape (batch, dim) or (dim,) for position and momentum.
    dt:
        Time step.

    Returns
    -------
    q_next, p_next:
        Updated position and momentum tensors (detached from graph).
    """
    # Ensure batch-dimension consistency
    if q.shape != p.shape:
        raise ValueError(f"q and p must have same shape, got {q.shape} vs {p.shape}")

    # First half-step in momentum at t_n
    dq_dt, dp_dt = hnn_time_derivatives(model, q, p)
    p_half = p + 0.5 * dt * dp_dt

    # Full step in position using p_{n+1/2}
    dq_dt_half, _ = hnn_time_derivatives(model, q, p_half)
    q_next = q + dt * dq_dt_half

    # Second half-step in momentum at t_{n+1}
    _, dp_dt_next = hnn_time_derivatives(model, q_next, p_half)
    p_next = p_half + 0.5 * dt * dp_dt_next

    # Detach so the next step starts from a fresh graph
    return q_next.detach(), p_next.detach()


def integrate_leapfrog(
    model: HamiltonianNN,
    y0: Tensor,
    t0: float,
    t1: float,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """
    Integrate HNN dynamics with Leapfrog from t0 to t1.

    Parameters
    ----------
    model:
        Trained HamiltonianNN instance.
    y0:
        Initial state tensor of shape (2 * dim,) or (1, 2 * dim),
        concatenated as [q, p].
    t0, t1:
        Start and end times.
    dt:
        Integration time step.

    Returns
    -------
    times:
        1D tensor of shape (T,) with time stamps.
    states:
        2D tensor of shape (T, 2 * dim) with concatenated [q(t), p(t)].
    """
    model.eval()

    if y0.dim() == 1:
        y = y0.unsqueeze(0)  # (1, 2*dim)
    elif y0.dim() == 2 and y0.shape[0] == 1:
        y = y0.clone()
    else:
        raise ValueError(f"y0 must be (2*dim,) or (1, 2*dim), got {y0.shape}")

    batch, state_dim = y.shape
    if batch != 1:
        raise ValueError("integrate_leapfrog currently supports batch size 1 only.")

    dim = state_dim // 2
    q, p = y[:, :dim], y[:, dim:]

    times: List[float] = []
    states: List[Tensor] = []

    t = float(t0)
    t_final = float(t1)
    dt = float(dt)

    while t <= t_final + 1e-12:
        times.append(t)
        states.append(torch.cat([q, p], dim=1).detach())

        # One Leapfrog step
        q, p = leapfrog_step(model, q, p, dt)
        t += dt

    times_tensor = torch.tensor(times, dtype=torch.float32)
    states_tensor = torch.vstack(states)  # (T, 2*dim)

    return times_tensor, states_tensor
