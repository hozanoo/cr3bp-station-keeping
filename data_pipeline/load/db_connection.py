# data_pipeline/load/db_connection.py
"""
Database connection utilities for the CR3BP data pipelines.

This module centralizes configuration and connection handling for
PostgreSQL-based storage used by the HNN training data pipeline
(and optionally other mission databases).

Connections are configured via environment variables. Typical
variables for the HNN physics database are:

- HNN_DB_HOST
- HNN_DB_PORT
- HNN_DB_NAME
- HNN_DB_USER
- HNN_DB_PASSWORD

If a DSN string is preferred, HNN_DB_DSN can be set instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor


@dataclass
class DbConfig:
    """
    Simple container for PostgreSQL connection parameters.
    """
    host: str
    port: int
    dbname: str
    user: str
    password: str
    dsn: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "HNN_DB_") -> "DbConfig":
        """
        Build a DbConfig instance from environment variables.

        If a DSN is provided via ``<PREFIX>DSN``, it will be stored
        in the ``dsn`` field and used directly when connecting.

        Parameters
        ----------
        prefix:
            Environment variable prefix, e.g. ``"HNN_DB_"`` or
            another value for a different logical database.

        Returns
        -------
        DbConfig
            Parsed configuration.
        """
        dsn = os.getenv(f"{prefix}DSN")

        host = os.getenv(f"{prefix}HOST", "localhost")
        port_str = os.getenv(f"{prefix}PORT", "5432")
        dbname = os.getenv(f"{prefix}NAME", "db_hnn")
        user = os.getenv(f"{prefix}USER", "hnn_user")
        password = os.getenv(f"{prefix}PASSWORD", "")

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port in {prefix}PORT: {port_str!r}")

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            dsn=dsn,
        )


def get_connection(cfg: Optional[DbConfig] = None) -> PgConnection:
    """
    Open a new PostgreSQL connection using the given configuration.

    Parameters
    ----------
    cfg:
        Database configuration. If omitted, configuration is read
        from environment variables via :meth:`DbConfig.from_env`.

    Returns
    -------
    psycopg2.extensions.connection
        A new database connection instance.
    """
    if cfg is None:
        cfg = DbConfig.from_env()

    if cfg.dsn:
        conn = psycopg2.connect(cfg.dsn)
    else:
        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            dbname=cfg.dbname,
            user=cfg.user,
            password=cfg.password,
        )

    return conn


@contextmanager
def db_cursor(
    cfg: Optional[DbConfig] = None,
    autocommit: bool = True,
) -> Iterator[PgCursor]:
    """
    Context manager yielding a PostgreSQL cursor.

    A connection is created on entry and closed automatically on
    exit. Optionally, autocommit can be disabled to manage
    transactions manually.

    Parameters
    ----------
    cfg:
        Database configuration or ``None`` to use
        :meth:`DbConfig.from_env`.
    autocommit:
        Whether to enable autocommit on the connection.

    Yields
    ------
    psycopg2.extensions.cursor
        A cursor instance ready for executing SQL statements.
    """
    conn = get_connection(cfg)
    conn.autocommit = autocommit

    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()
