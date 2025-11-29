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
    Hamiltonian Neural Network that maps (q, p) -> H(q, p).

    The network learns a scalar Hamiltonian H, from which time derivatives
    are obtained via Hamilton's equations:

        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q
    """

    def __init__(
        self,
        dim: int = 3,
        hidden_dims: Iterable[int] | Tuple[int, ...] = (256, 256, 256),
        activation: str = "sine",
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.input_dim = 2 * self.dim
        self.hidden_dims = tuple(hidden_dims)

        self.net = make_mlp(
            in_dim=self.input_dim,
            out_dim=1,
            hidden_dims=self.hidden_dims,
            activation=activation,
        )

        # Normalization buffers over concatenated [q, p]
        self.register_buffer("state_mean", torch.zeros(self.input_dim))
        self.register_buffer("state_std", torch.ones(self.input_dim))

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
        """
        if mean.shape != self.state_mean.shape:
            raise ValueError(f"mean must have shape {self.state_mean.shape}, got {mean.shape}")
        if std.shape != self.state_std.shape:
            raise ValueError(f"std must have shape {self.state_std.shape}, got {std.shape}")

        with torch.no_grad():
            self.state_mean.copy_(mean)
            self.state_std.copy_(std.clamp_min(1e-6))

    def _standardize_state(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate q and p and apply feature-wise standardization.
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if p.dim() == 1:
            p = p.unsqueeze(0)

        state = torch.cat([q, p], dim=-1)
        norm = (state - self.state_mean) / self.state_std
        return norm

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar Hamiltonian H(q, p) for each sample.
        """
        x = self._standardize_state(q, p)
        H = self.net(x)
        return H.squeeze(-1)

    # ------------------------------------------------------------------
    # Time derivatives via Hamilton's equations
    # ------------------------------------------------------------------

    def time_derivatives(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dq/dt and dp/dt using Hamilton's equations.

        This method is robust against outer `torch.no_grad()` contexts by
        explicitly enabling gradients inside.
        """
        # Enforce batch dimension
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if p.dim() == 1:
            p = p.unsqueeze(0)

        # Enable gradients even if outer code disabled them
        with torch.enable_grad():
            # Local copies that definitely require gradients
            q_req = q.clone().detach().requires_grad_(True)
            p_req = p.clone().detach().requires_grad_(True)

            # Forward pass: build a computation graph for H
            H = self.forward(q_req, p_req)   # shape (batch,)
            H_sum = H.sum()                  # scalar

            # Gradients wrt q and p
            dH_dq, dH_dp = torch.autograd.grad(
                H_sum,
                (q_req, p_req),
                create_graph=True,
                retain_graph=True,
            )

            dq_dt = dH_dp
            dp_dt = -dH_dq

        return dq_dt, dp_dt
