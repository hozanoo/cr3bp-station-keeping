# hnn_models/dataloader/hnn_dataset.py
"""
PyTorch Dataset for CR3BP HNN training.

The dataset reads samples from the PostgreSQL view ``hnn_training_view``
and groups them by episode. Positions, velocities and accelerations are
taken directly from the database.

Each item corresponds to a single time step and contains:
- q      : position vector (dim,)
- p      : velocity vector (dim,)
- dq_dt  : ground-truth time derivative of q  (equals velocity)
- dp_dt  : ground-truth time derivative of p  (acceleration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    # Use the shared CR3BP DB prefix (DB_HOST, DB_PORT, ...)
    db_env_prefix: str = "DB_"
    where_clause: Optional[str] = None
    limit: Optional[int] = None
    dtype: np.dtype = np.float32


class HnnTrainingDataset(Dataset):
    """
    Dataset providing CR3BP samples for Hamiltonian NN training.

    Data is loaded eagerly from PostgreSQL at construction time.
    """

    def __init__(self, config: Optional[HnnDatasetConfig] = None) -> None:
        super().__init__()
        self.config = config or HnnDatasetConfig()

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
        cfg = DbConfig.from_env(prefix_env=self.config.db_env_prefix)

        base_query = """
            SELECT
                episode_id,
                dim,
                t,
                x, y, z,
                vx, vy, vz,
                ax, ay, az
            FROM hnn_training_view
        """

        where_clauses: List[str] = []
        params: List[object] = []

        # Filter by dimensionality (2D vs 3D)
        if self.config.dim in (2, 3):
            where_clauses.append("dim = %s")
            params.append(self.config.dim)
        else:
            raise ValueError(f"Unsupported dimension: {self.config.dim!r}")

        if self.config.where_clause:
            where_clauses.append(f"({self.config.where_clause})")

        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)

        base_query += " ORDER BY episode_id, t"

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

        q_list: List[np.ndarray] = []
        p_list: List[np.ndarray] = []
        dq_dt_list: List[np.ndarray] = []
        dp_dt_list: List[np.ndarray] = []

        for episode_id, group in df.groupby("episode_id", sort=True):
            group = group.sort_values("t")

            if self.config.dim == 3:
                pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
                vel = group[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
                acc = group[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
            else:
                pos = group[["x", "y"]].to_numpy(dtype=np.float64)
                vel = group[["vx", "vy"]].to_numpy(dtype=np.float64)
                acc = group[["ax", "ay"]].to_numpy(dtype=np.float64)

            q_list.append(pos)
            p_list.append(vel)
            dq_dt_list.append(vel)
            dp_dt_list.append(acc)

        q = np.concatenate(q_list, axis=0).astype(self.config.dtype, copy=False)
        p = np.concatenate(p_list, axis=0).astype(self.config.dtype, copy=False)
        dq_dt = np.concatenate(dq_dt_list, axis=0).astype(self.config.dtype, copy=False)
        dp_dt = np.concatenate(dp_dt_list, axis=0).astype(self.config.dtype, copy=False)

        state = np.concatenate([q, p], axis=1)
        state_mean, state_std = compute_standardization_stats(state)

        return q, p, dq_dt, dp_dt, state_mean, state_std
