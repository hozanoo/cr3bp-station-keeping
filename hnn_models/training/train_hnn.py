# hnn_models/training/train_hnn.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from hnn_models.dataloader.hnn_dataset import HnnDatasetConfig, HnnTrainingDataset
from hnn_models.model.hnn import HamiltonianNN


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    Configuration dataclass for HNN training.

    Attributes
    ----------
    dim : int
        Dimensionality of the problem (e.g., 2 or 3).
    hidden_dims : Tuple[int, ...]
        Structure of the hidden layers in the MLP.
    batch_size : int
        Size of the training batches.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the Adam optimizer.
    limit : Optional[int]
        Limit the number of samples loaded from the database.
    where_clause : Optional[str]
        SQL WHERE clause to filter training data (e.g., for excluding crashes).
    run_name : str
        Name of the run, used for checkpoint filenames.
    load_checkpoint : Optional[str]
        Path to a .pt file to resume training or fine-tune.
    """
    dim: int = 3
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    batch_size: int = 1024
    epochs: int = 30
    lr: float = 2e-4
    limit: Optional[int] = None
    where_clause: Optional[str] = None
    run_name: str = "hnn_cr3bp_run"
    load_checkpoint: Optional[str] = None


# ----------------------------------------------------------------------
# Loss computation
# ----------------------------------------------------------------------


def compute_loss(
    model: HamiltonianNN,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute the physics-informed loss for a single batch.

    The loss combines the mean squared error of the time derivatives
    (dq/dt and dp/dt). It normalizes the error by the batch-wise RMS
    scale of the ground truth to ensure balanced gradients.

    Parameters
    ----------
    model : HamiltonianNN
        The neural network model.
    batch : tuple
        A tuple containing (q, p, dq_true, dp_true).
    criterion : nn.Module
        The loss function (e.g., MSELoss).
    device : torch.device
        The compute device (CPU or GPU).

    Returns
    -------
    tuple[torch.Tensor, float, float]
        Total loss (Tensor), dq loss (float), dp loss (float).
    """
    q, p, dq_true, dp_true = batch

    q = q.to(device)
    p = p.to(device)
    dq_true = dq_true.to(device)
    dp_true = dp_true.to(device)

    # Model calculates derivatives internally via autograd
    dq_pred, dp_pred = model.time_derivatives(q, p)

    eps = 1e-8
    with torch.no_grad():
        scale_dq = dq_true.pow(2).mean().sqrt().clamp_min(eps)
        scale_dp = dp_true.pow(2).mean().sqrt().clamp_min(eps)

    dq_pred_n = dq_pred / scale_dq
    dq_true_n = dq_true / scale_dq
    dp_pred_n = dp_pred / scale_dp
    dp_true_n = dp_true / scale_dp

    loss_dq = criterion(dq_pred_n, dq_true_n)
    loss_dp = criterion(dp_pred_n, dp_true_n)

    loss = loss_dq + loss_dp

    return loss, float(loss_dq.detach().cpu()), float(loss_dp.detach().cpu())


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------


def train_hnn(config: TrainConfig) -> None:
    """
    Execute the main training loop for the Hamiltonian Neural Network.

    This function handles:
    1. Dataset loading and splitting.
    2. Model initialization (or checkpoint loading).
    3. State normalization logic (preserving stats if fine-tuning).
    4. Training loop with gradient clipping.
    5. Validation and checkpoint saving.

    Parameters
    ----------
    config : TrainConfig
        Training configuration object.
    """
    # Force CPU to ensure consistency with double precision requirements often found in scientific computing
    device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset Initialization
    ds_cfg = HnnDatasetConfig(
        dim=config.dim,
        where_clause=config.where_clause,
        limit=config.limit,
    )
    dataset = HnnTrainingDataset(ds_cfg)
    n_total = len(dataset)
    print(f"[INFO] Dataset size: {n_total} (limit={config.limit})")

    if n_total == 0:
        raise RuntimeError("Empty dataset. Check where_clause and limit.")

    # Train/validation split (90/10)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    workers=0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True
    )

    # Model Initialization
    model = HamiltonianNN(dim=config.dim, hidden_dims=config.hidden_dims, activation="sine")
    model.to(device)

    # Checkpoint Loading
    if config.load_checkpoint:
        print(f"[INFO] Loading weights from checkpoint: {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # State Normalization Logic
    # If loading a checkpoint (fine-tuning), we typically want to preserve the
    # normalization statistics of the original model to maintain the learned manifold.
    if not config.load_checkpoint:
        print("[INFO] Setting state normalization from current dataset stats.")
        model.set_state_normalization(
            torch.from_numpy(dataset.state_mean),
            torch.from_numpy(dataset.state_std),
        )
    else:
        print("[INFO] Checkpoint loaded: Keeping existing normalization stats for fine-tuning.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Checkpoint Directory Setup
    this_file = Path(__file__).resolve()
    checkpoints_dir = this_file.parents[1] / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"{config.run_name}.pt"
    meta_path = checkpoints_dir / f"{config.run_name}_meta.json"

    best_val_loss = float("inf")

    # Epoch Loop
    for epoch in range(1, config.epochs + 1):
        # ----------------------------
        # Training Phase
        # ----------------------------
        model.train()
        train_losses: List[float] = []
        train_dq_losses: List[float] = []
        train_dp_losses: List[float] = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dq, loss_dp = compute_loss(model, batch, criterion, device)
            loss.backward()

            # Gradient Clipping: Prevents exploding gradients from high-acceleration samples
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            train_dq_losses.append(loss_dq)
            train_dp_losses.append(loss_dp)

        train_loss = float(np.mean(train_losses))
        train_dq_loss = float(np.mean(train_dq_losses))
        train_dp_loss = float(np.mean(train_dp_losses))

        # ----------------------------
        # Validation Phase
        # ----------------------------
        model.eval()
        val_losses: List[float] = []
        val_dq_losses: List[float] = []
        val_dp_losses: List[float] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False):
                loss, loss_dq, loss_dp = compute_loss(model, batch, criterion, device)
                val_losses.append(float(loss.detach().cpu()))
                val_dq_losses.append(loss_dq)
                val_dp_losses.append(loss_dp)

        val_loss = float(np.mean(val_losses))
        val_dq_loss = float(np.mean(val_dq_losses))
        val_dp_loss = float(np.mean(val_dp_losses))

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.6e} "
            f"val_loss={val_loss:.6e} "
            f"(dq: {val_dq_loss:.6e}, dp: {val_dp_loss:.6e})"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

            meta = {
                "dim": config.dim,
                "hidden_dims": list(config.hidden_dims),
                "state_mean": dataset.state_mean.tolist(),
                "state_std": dataset.state_std.tolist(),
                "best_val_loss": best_val_loss,
                "run_name": config.run_name,
                "limit": config.limit,
                "where_clause": config.where_clause,
                "parent_checkpoint": config.load_checkpoint,
            }
            with meta_path.open("w", encoding="utf8") as f:
                json.dump(meta, f, indent=2)

            print(f"[INFO] Saved new best checkpoint to {ckpt_path}")

    print("[INFO] Training finished.")
    print(f"[INFO] Best validation loss: {best_val_loss:.6e}")


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CR3BP Hamiltonian Neural Network.")

    parser.add_argument("--dim", type=int, default=3, help="Spatial dimensionality (2 or 3)")
    
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 512], help="Hidden layer sizes")
    
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples to load")
    parser.add_argument("--where-clause", type=str, default=None, help="SQL filter for dataset")
    parser.add_argument("--run-name", type=str, default="hnn_cr3bp_run", help="Name of the run")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to .pt checkpoint to resume from")

    args = parser.parse_args()

    cfg = TrainConfig(
        dim=args.dim,
        hidden_dims=tuple(args.hidden_dims),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        limit=args.limit,
        where_clause=args.where_clause,
        run_name=args.run_name,
        load_checkpoint=args.load_checkpoint,
    )
    return cfg


if __name__ == "__main__":
    print("DEBUG: Skript gestartet...") 
    
    config = parse_args()
    
    print(f"DEBUG: Config geladen. Starte Training mit {config.limit} Datenpunkten...")
    
    train_hnn(config)