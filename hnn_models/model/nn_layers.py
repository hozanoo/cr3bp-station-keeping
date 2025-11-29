# hnn_models/model/nn_layers.py
"""
Utility functions for constructing neural network components.
Supports sine, tanh, relu activations.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """
    Return an activation module given its name.

    Supported:
    - "tanh"
    - "relu"
    - "sine"   (SIREN-style)
    """
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sine":
        # SIREN-like sine activation
        class Sine(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(x)

        return Sine()

    raise ValueError(f"Unsupported activation function: {name}")


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Iterable[int] | Tuple[int, ...],
    activation: str = "tanh",
) -> nn.Sequential:
    """
    Build a fully-connected feed-forward neural network (MLP).

    Parameters
    ----------
    in_dim:
        Number of input features.
    out_dim:
        Number of output features.
    hidden_dims:
        Sizes of hidden layers.
    activation:
        Activation function to use in all hidden layers.

    Returns
    -------
    nn.Sequential
        A feed-forward network: Linear + Activation blocks.
    """
    layers = []
    current_dim = in_dim

    act = get_activation(activation)

    for h in hidden_dims:
        layers.append(nn.Linear(current_dim, h))
        layers.append(act)
        current_dim = h

    # Final linear layer (no activation)
    layers.append(nn.Linear(current_dim, out_dim))

    return nn.Sequential(*layers)
