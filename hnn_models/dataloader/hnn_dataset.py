# hnn_models/dataloader/hnn_dataset.py
"""
PyTorch Dataset for CR3BP HNN training.

This dataset reads samples from the PostgreSQL view ``hnn_training_view``.
The view is expected to provide, at minimum, the following columns:

    episode_id, dim, t,
    x, y, z,
    vx, vy, vz,
    ax, ay, az

The dataset performs the following steps:

* Selects rows for the requested spatial dimension ``dim``.
* Builds position vectors q = (x, y) or (x, y, z).
* Builds **canonical momenta** p, not raw velocities:

    In the rotating CR3BP frame (normalized units):

        p_x = v_x - y
        p_y = v_y + x
        p_z = v_z  (for dim = 3)

* Uses as ground-truth derivatives:

        dq_dt = v   (physical velocity)
        dp_dt = dp/dt, derived from:

            p_x = v_x - y  →  dp_x/dt = a_x - v_y
            p_y = v_y + x  →  dp_y/dt = a_y + v_x
            p_z = v_z      →  dp_z/dt = a_z

* Computes mean and std of the concatenated state [q, p] for standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.load.db_connection import DbConfig, db_cursor
from hnn_models.dataloader.preprocessing import compute_standardization_stats


@dataclass
class HnnDatasetConfig:
    """
    Configuration for the CR3BP HNN training dataset.
    """
    dim: int = 3
    # Environment variable prefix for DB connection
    db_env_prefix: str = "DB_"
    # Optional SQL WHERE clause appended to the view query
    where_clause: Optional[str] = None
    # Optional hard limit on number of samples
    limit: Optional[int] = None
    # Numpy dtype for stored arrays
    dtype: np.dtype = np.float32


class HnnTrainingDataset(Dataset):
    """
    Dataset providing CR3BP samples for Hamiltonian NN training.

    Each item is a single time step and consists of:

        q      : position vector          (dim,)
        p      : canonical momentum       (dim,)
        dq_dt  : time derivative of q     (dim,)  (equals velocity)
        dp_dt  : time derivative of p     (dim,)  (derived from a and v)

    All internal arrays are stored as numpy arrays with dtype given
    in the configuration (float32 by default).
    """

    def __init__(self, config: HnnDatasetConfig) -> None:
        super().__init__()
        self.config = config

        (
            self._q,
            self._p,
            self._dq_dt,
            self._dp_dt,
            self.state_mean,
            self.state_std,
        ) = self._load_from_db()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._q.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.from_numpy(self._q[idx])
        p = torch.from_numpy(self._p[idx])
        dq_dt = torch.from_numpy(self._dq_dt[idx])
        dp_dt = torch.from_numpy(self._dp_dt[idx])
        return q, p, dq_dt, dp_dt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load samples from the PostgreSQL view ``hnn_training_view``.

        Returns
        -------
        q, p, dq_dt, dp_dt, state_mean, state_std
            All arrays have shape (N, dim), where dim is the configured
            spatial dimension. state_mean and state_std have shape
            (2 * dim,) and correspond to the concatenated state [q, p].
        """
        cfg = DbConfig.from_env(prefix_env=self.config.db_env_prefix)

        base_query = """
            SELECT
                episode_id,
                dim,
                t,
                x, y, z,
                vx, vy, vz,
                ax, ay, az
            FROM cr3bp.hnn_training_view
        """
        clauses = []
        params: list = []

        if self.config.where_clause:
            clauses.append(self.config.where_clause)

        # Always enforce the requested dimension
        clauses.append("dim = %s")
        params.append(int(self.config.dim))

        if clauses:
            base_query += " WHERE " + " AND ".join(f"({c})" for c in clauses)

        if self.config.limit is not None:
            base_query += f" LIMIT {int(self.config.limit)}"

        with db_cursor(cfg) as cur:
            cur.execute(base_query, params)
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError("No rows returned from hnn_training_view for the given configuration.")

        columns = [
            "episode_id",
            "dim",
            "t",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "ax",
            "ay",
            "az",
        ]
        df = pd.DataFrame(rows, columns=columns)

        # Positions
        if self.config.dim == 3:
            q = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
            v = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
            a = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
        elif self.config.dim == 2:
            q = df[["x", "y"]].to_numpy(dtype=np.float64)
            v = df[["vx", "vy"]].to_numpy(dtype=np.float64)
            a = df[["ax", "ay"]].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Unsupported dim={self.config.dim} in HnnTrainingDataset.")

        # Canonical momenta in rotating CR3BP frame:
        #   p_x = v_x - y
        #   p_y = v_y + x
        #   p_z = v_z (for dim=3)
        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)

        if self.config.dim == 3:
            vx = df["vx"].to_numpy(dtype=np.float64)
            vy = df["vy"].to_numpy(dtype=np.float64)
            vz = df["vz"].to_numpy(dtype=np.float64)

            p_x = vx - y
            p_y = vy + x
            p_z = vz

            p = np.stack([p_x, p_y, p_z], axis=1)

            ax = df["ax"].to_numpy(dtype=np.float64)
            ay = df["ay"].to_numpy(dtype=np.float64)
            az = df["az"].to_numpy(dtype=np.float64)

            dp_x = ax - vy
            dp_y = ay + vx
            dp_z = az

            dp_dt = np.stack([dp_x, dp_y, dp_z], axis=1)
        else:
            vx = df["vx"].to_numpy(dtype=np.float64)
            vy = df["vy"].to_numpy(dtype=np.float64)

            p_x = vx - y
            p_y = vy + x
            p = np.stack([p_x, p_y], axis=1)

            ax = df["ax"].to_numpy(dtype=np.float64)
            ay = df["ay"].to_numpy(dtype=np.float64)

            dp_x = ax - vy
            dp_y = ay + vx
            dp_dt = np.stack([dp_x, dp_y], axis=1)

        # dq/dt is simply the physical velocity
        dq_dt = v

        # Cast to configured dtype
        q = q.astype(self.config.dtype, copy=False)
        p = p.astype(self.config.dtype, copy=False)
        dq_dt = dq_dt.astype(self.config.dtype, copy=False)
        dp_dt = dp_dt.astype(self.config.dtype, copy=False)

        # Standardization stats on concatenated state
        state = np.concatenate([q, p], axis=1)
        state_mean, state_std = compute_standardization_stats(state)

        return q, p, dq_dt, dp_dt, state_mean, state_std
