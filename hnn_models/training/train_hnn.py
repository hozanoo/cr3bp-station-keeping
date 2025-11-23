# hnn_models/training/train_hnn.py
"""
Training script for the CR3BP Hamiltonian Neural Network.

This module can be used as a library function or as a script:

    python -m hnn_models.training.train_hnn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from hnn_models.dataloader.hnn_dataset import HnnTrainingDataset, HnnDatasetConfig
from hnn_models.model.hnn import HamiltonianNN


def create_dataloader(
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    dim: int = 3,
) -> Tuple[HnnTrainingDataset, DataLoader]:
    """
    Construct the HNN training dataset and dataloader.
    """
    dataset = HnnTrainingDataset(HnnDatasetConfig(dim=dim))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader


def train_hnn(
    dim: int = 3,
    hidden_dims = (128, 128, 128),
    batch_size: int = 1024,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_dq: float = 1.0,
    weight_dp: float = 1.0,
    output_dir: Path | str = Path("hnn_models/checkpoints"),
) -> Path:
    """
    Train a Hamiltonian NN on CR3BP trajectories.

    Parameters
    ----------
    dim:
        Spatial dimensionality (2 or 3).
    hidden_dims:
        Hidden layer sizes for the MLP energy network.
    batch_size:
        Training batch size.
    num_epochs:
        Number of passes over the dataset.
    learning_rate:
        Optimizer learning rate.
    weight_dq, weight_dp:
        Relative weights for dq/dt and dp/dt components in the loss.
    output_dir:
        Directory where model checkpoints and metadata will be stored.

    Returns
    -------
    ckpt_path:
        Path to the final saved model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader = create_dataloader(
        batch_size=batch_size,
        dim=dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HamiltonianNN(dim=dim, hidden_dims=hidden_dims)
    state_mean = torch.from_numpy(dataset.state_mean).to(torch.float32)
    state_std = torch.from_numpy(dataset.state_std).to(torch.float32)
    model.set_state_normalization(state_mean, state_std)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for q, p, dq_dt_true, dp_dt_true in loader:
            q = q.to(device=device, dtype=torch.float32)
            p = p.to(device=device, dtype=torch.float32)
            dq_dt_true = dq_dt_true.to(device=device, dtype=torch.float32)
            dp_dt_true = dp_dt_true.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            dq_dt_pred, dp_dt_pred = model.time_derivatives(q, p)

            loss_dq = criterion(dq_dt_pred, dq_dt_true)
            loss_dp = criterion(dp_dt_pred, dp_dt_true)
            loss = weight_dq * loss_dq + weight_dp * loss_dp

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6e}")

    ckpt_path = output_dir / "hnn_cr3bp_l1.pt"
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "dim": dim,
        "hidden_dims": list(hidden_dims),
        "state_mean": dataset.state_mean.tolist(),
        "state_std": dataset.state_std.tolist(),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
    }
    meta_path = output_dir / "hnn_cr3bp_l1_meta.json"
    with meta_path.open("w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved HNN checkpoint to {ckpt_path}")
    print(f"Saved HNN metadata to    {meta_path}")

    return ckpt_path


if __name__ == "__main__":
    train_hnn()
