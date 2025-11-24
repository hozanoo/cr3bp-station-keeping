# hnn_models/training/test_leapfrog_rollout.py
"""
Quick sanity check: HNN rollout with Leapfrog integrator.

Usage:
    python -m hnn_models.training.test_leapfrog_rollout
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hnn_models.dataloader.hnn_dataset import HnnTrainingDataset, HnnDatasetConfig
from hnn_models.model.hnn import HamiltonianNN
from hnn_models.utilities.integration import integrate_leapfrog


def main() -> None:
    device = torch.device("cpu")

    # Load dataset to get a realistic initial state
    dataset = HnnTrainingDataset(HnnDatasetConfig(dim=3))
    q0, p0, _, _ = dataset[0]
    y0 = torch.cat([q0, p0]).to(dtype=torch.float32, device=device)

    # Load metadata + model weights
    ckpt_dir = Path("hnn_models/checkpoints")
    ckpt_path = ckpt_dir / "hnn_cr3bp_l1.pt"
    meta_path = ckpt_dir / "hnn_cr3bp_l1_meta.json"

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    dim = meta["dim"]
    hidden_dims = meta["hidden_dims"]
    state_mean = torch.tensor(meta["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(meta["state_std"], dtype=torch.float32, device=device)

    model = HamiltonianNN(dim=dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_state_normalization(state_mean, state_std)
    model.to(device)

    # Run a short Leapfrog rollout
    t0, t1, dt = 0.0, 2.0, 0.01
    times, states = integrate_leapfrog(model, y0, t0, t1, dt)

    print("Rollout finished.")
    print("times.shape :", times.shape)
    print("states.shape:", states.shape)


if __name__ == "__main__":
    main()
