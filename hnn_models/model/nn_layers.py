# hnn_models/model/nn_layers.py
"""
Neural network building blocks for HNN models.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str = "tanh",
) -> nn.Sequential:
    """
    Construct a simple fully-connected MLP.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    hidden_dims:
        Iterable of hidden layer sizes.
    output_dim:
        Output dimension.
    activation:
        Name of activation function ("tanh", "relu", "gelu").

    Returns
    -------
    model:
        ``nn.Sequential`` implementing the MLP.
    """
    hidden_dims = list(hidden_dims)
    layers: List[nn.Module] = []

    act_cls: nn.Module
    if activation == "tanh":
        act_cls = nn.Tanh
    elif activation == "relu":
        act_cls = nn.ReLU
    elif activation == "gelu":
        act_cls = nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation!r}")

    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_cls())
        prev_dim = h

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)
