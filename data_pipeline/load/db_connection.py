"""
Database connection utilities for the CR3BP data pipelines.

This module centralizes configuration and connection handling for
PostgreSQL-based storage used by the CR3BP project
(e.g. Airflow loaders, HNN training data, simulation exports).

Connections are configured via environment variables. The *default*
convention in this project is a generic CR3BP database:

- DB_HOST
- DB_PORT
- DB_NAME
- DB_USER
- DB_PASSWORD

Optionally, a DSN or URL can be used:

- CR3BP_DATABASE_URL   (e.g. postgresql+psycopg2://user:pass@host:5432/db)
- <PREFIX>DSN          (e.g. DB_DSN or HNN_DB_DSN)

For special cases (e.g. a separate HNN-only database) you can still
use a different prefix like::

    DbConfig.from_env(
        prefix_env="HNN_DB_",
        url_env="HNN_DB_URL",
        default_dbname="db_hnn",
        default_user="hnn_user",
    )

By default, this module is configured to talk to the shared CR3BP
database inside Docker (service name "postgres").
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
    def from_env(
        cls,
        prefix_env: str = "DB_",
        url_env: str = "CR3BP_DATABASE_URL",
        default_host: str = "localhost",
        default_port: int = 5432,
        default_dbname: str = "cr3bp_db",
        default_user: str = "cr3bp_user",
        default_password: str = "",
    ) -> "DbConfig":
        """
        Build a DbConfig instance from environment variables.

        Resolution order:

        1. If ``url_env`` (e.g. ``CR3BP_DATABASE_URL``) is set:
           use it as base URL/DSN.
        2. Else, if ``<PREFIX>DSN`` (e.g. ``DB_DSN``) is set:
           use that.
        3. Else, fall back to individual fields::

               <PREFIX>HOST
               <PREFIX>PORT
               <PREFIX>NAME
               <PREFIX>USER
               <PREFIX>PASSWORD

        Parameters
        ----------
        prefix_env:
            Prefix for individual env vars (default: ``"DB_"``).
        url_env:
            Name of an environment variable that contains a full
            database URL/DSN (default: ``"CR3BP_DATABASE_URL"``).
        default_host, default_port, default_dbname, default_user, default_password:
            Fallback values when no environment variables are set.

        Returns
        -------
        DbConfig
            Parsed configuration.
        """
        # 1) URL-style DSN (e.g. CR3BP_DATABASE_URL or HNN_DB_URL)
        raw_dsn = os.getenv(url_env)

        # 2) Fallback: prefix-based DSN (e.g. DB_DSN or HNN_DB_DSN)
        if not raw_dsn:
            raw_dsn = os.getenv(f"{prefix_env}DSN")

        dsn: Optional[str] = None
        if raw_dsn:
            # psycopg2 does not understand "+psycopg2" in the scheme
            if "+psycopg2" in raw_dsn:
                raw_dsn = raw_dsn.replace("+psycopg2", "")
            dsn = raw_dsn

        host = os.getenv(f"{prefix_env}HOST", default_host)
        port_str = os.getenv(f"{prefix_env}PORT", str(default_port))
        dbname = os.getenv(f"{prefix_env}NAME", default_dbname)
        user = os.getenv(f"{prefix_env}USER", default_user)
        password = os.getenv(f"{prefix_env}PASSWORD", default_password)

        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(f"Invalid port in {prefix_env}PORT: {port_str!r}") from exc

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
