# hnn_models/model/hnn.py
"""
Hamiltonian Neural Network for CR3BP dynamics.

The model learns a scalar Hamiltonian H(q, p) from which time
derivatives are obtained via Hamilton's equations:

    dq/dt =  ∂H/∂p
    dp/dt = -∂H/∂q
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

from hnn_models.model.nn_layers import make_mlp


class HamiltonianNN(nn.Module):
    """
    Simple Hamiltonian Neural Network with an MLP energy function.

    The network takes concatenated (q, p) as input and produces
    a scalar Hamiltonian value. Time derivatives are obtained via
    automatic differentiation.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Iterable[int] = (128, 128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        input_dim = 2 * self.dim

        self.energy_net = make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

        self.register_buffer("state_mean", torch.zeros(input_dim))
        self.register_buffer("state_std", torch.ones(input_dim))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def set_state_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """
        Store feature-wise mean and std for input normalization.

        Parameters
        ----------
        mean:
            Tensor of shape (2 * dim,) with state mean.
        std:
            Tensor of shape (2 * dim,) with state std (clipped to > 0).
        """
        if mean.shape != self.state_mean.shape:
            raise ValueError(f"mean must have shape {self.state_mean.shape}, got {mean.shape}")
        if std.shape != self.state_std.shape:
            raise ValueError(f"std must have shape {self.state_std.shape}, got {std.shape}")

        with torch.no_grad():
            self.state_mean.copy_(mean)
            self.state_std.copy_(std)

    # ------------------------------------------------------------------
    # Core forward and dynamics
    # ------------------------------------------------------------------

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian H(q, p).

        Parameters
        ----------
        q:
            Position tensor of shape (..., dim).
        p:
            Momentum/velocity tensor of shape (..., dim).

        Returns
        -------
        H:
            Tensor of shape (..., 1) with Hamiltonian values.
        """
        if q.shape != p.shape:
            raise ValueError(f"q and p must have the same shape, got {q.shape} vs {p.shape}")

        state = torch.cat([q, p], dim=-1)
        state_norm = (state - self.state_mean) / self.state_std
        H = self.energy_net(state_norm)
        return H

    def time_derivatives(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time derivatives dq/dt and dp/dt via Hamilton's equations.

        Both q and p are treated as differentiable variables.

        Parameters
        ----------
        q, p:
            Tensors of shape (batch, dim) or (dim,) representing positions
            and velocities/momenta.

        Returns
        -------
        dq_dt, dp_dt:
            Tensors with the same shape as q and p.
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)

        H = self.forward(q, p)
        H_sum = H.sum()

        dH_dq, dH_dp = torch.autograd.grad(
            H_sum,
            (q, p),
            create_graph=True,
            retain_graph=True,
        )

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return dq_dt, dp_dt
