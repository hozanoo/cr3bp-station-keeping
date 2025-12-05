# hnn_models/training/train_hnn.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    # Path to an existing meta.json file whose normalization should be reused
    norm_source_meta: Optional[str] = None


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
    Compute the scaled MSE loss for dq and dp.

    We normalize dq and dp by their batch RMS to keep the losses roughly
    on a similar scale across different regions of the state space.
    """
    q, p, dq_true, dp_true = batch

    q = q.to(device)
    p = p.to(device)
    dq_true = dq_true.to(device)
    dp_true = dp_true.to(device)

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
    Main training routine for the CR3BP Hamiltonian Neural Network.

    Normalization priority:
    1) If norm_source_meta is given: force state_mean/state_std from that meta.json.
    2) Else, if load_checkpoint is given: reuse normalization from checkpoint.
    3) Else: compute fresh normalization from the current dataset.
    """
    device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset initialization
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

    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Model initialization
    model = HamiltonianNN(dim=config.dim, hidden_dims=config.hidden_dims, activation="sine")
    model.to(device)

    # ------------------------------------------------------------------
    # Normalization strategy
    # ------------------------------------------------------------------
    # Priority 1: explicit normalization from external meta.json
    if config.norm_source_meta:
        norm_path = Path(config.norm_source_meta)
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization meta file not found: {norm_path}")

        print(f"[INFO] Forcing normalization from external file: {norm_path}")
        with norm_path.open("r", encoding="utf8") as f:
            meta_data = json.load(f)

        mean_loaded = torch.tensor(meta_data["state_mean"], dtype=torch.float32)
        std_loaded = torch.tensor(meta_data["state_std"], dtype=torch.float32)
        model.set_state_normalization(mean_loaded, std_loaded)

    # Priority 2: load checkpoint and reuse its normalization
    elif config.load_checkpoint:
        print(f"[INFO] Loading checkpoint (preserving its normalization): {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location=device)
        # This also restores registered buffers such as state_mean/state_std.
        model.load_state_dict(state_dict, strict=False)

    # Priority 3: fresh normalization from current dataset
    else:
        print("[INFO] Computing fresh normalization statistics from current dataset.")
        model.set_state_normalization(
            torch.from_numpy(dataset.state_mean),
            torch.from_numpy(dataset.state_std),
        )

    # If both a checkpoint and an external normalization are specified:
    # - First set normalization from meta,
    # - Then load weights from checkpoint,
    # - Then enforce the external normalization again in case it was overwritten.
    if config.load_checkpoint and config.norm_source_meta:
        print(f"[INFO] Loading weights from: {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.set_state_normalization(mean_loaded, std_loaded)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Output paths
    this_file = Path(__file__).resolve()
    checkpoints_dir = this_file.parents[1] / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"{config.run_name}.pt"
    meta_path = checkpoints_dir / f"{config.run_name}_meta.json"

    best_val_loss = float("inf")

    # Epoch loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        train_dq_losses = []
        train_dp_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dq, loss_dp = compute_loss(model, batch, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            train_dq_losses.append(loss_dq)
            train_dp_losses.append(loss_dp)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False):
                loss, _, _ = compute_loss(model, batch, criterion, device)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"[EPOCH {epoch:03d}] train={train_loss:.6e}  val={val_loss:.6e}")

        # Save best model + meta
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

            # IMPORTANT: always store the normalization actually used by the model
            meta = {
                "dim": config.dim,
                "hidden_dims": list(config.hidden_dims),
                "state_mean": model.state_mean.cpu().tolist(),
                "state_std": model.state_std.cpu().tolist(),
                "best_val_loss": best_val_loss,
                "run_name": config.run_name,
                "limit": config.limit,
                "where_clause": config.where_clause,
                "parent_checkpoint": config.load_checkpoint,
                "forced_norm_source": config.norm_source_meta,
            }
            with meta_path.open("w", encoding="utf8") as f:
                json.dump(meta, f, indent=2)

            print(f"[INFO] Saved best model to {ckpt_path}")

    print("[INFO] Training finished.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CR3BP Hamiltonian Neural Network.")

    parser.add_argument("--dim", type=int, default=3, help="Spatial dimensionality (2 or 3).")
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[512, 512, 512],
        help="Hidden layer sizes.",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to load from the dataset.",
    )
    parser.add_argument(
        "--where-clause",
        type=str,
        default=None,
        help="SQL WHERE clause used to filter the dataset.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="hnn_cr3bp_run",
        help="Name of the training run (used for checkpoint filenames).",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to initialize weights from.",
    )
    parser.add_argument(
        "--norm-source-meta",
        type=str,
        default=None,
        help="Path to a meta.json file whose state_mean/state_std should be reused.",
    )

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
        norm_source_meta=args.norm_source_meta,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train_hnn(config)
