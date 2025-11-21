"""
Database connection utilities for the CR3BP project.

This module centralises the logic for creating SQLAlchemy engines
and database sessions. All other modules that need to talk to
PostgreSQL should import their engine from here.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


DEFAULT_DB_URL = (
    "postgresql+psycopg2://cr3bp_user:cr3bp_password@postgres:5432/cr3bp_db"
)


def get_database_url(env_var: str = "CR3BP_DATABASE_URL") -> str:
    """
    Resolve the database URL from an environment variable.

    Parameters
    ----------
    env_var:
        Name of the environment variable that stores the database URL.
        If not set, a sensible default for local Docker development is used.

    Returns
    -------
    str
        A SQLAlchemy-compatible database URL.
    """
    return os.getenv(env_var, DEFAULT_DB_URL)


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    """
    Create a SQLAlchemy engine for the CR3BP PostgreSQL database.

    Parameters
    ----------
    db_url:
        Optional explicit database URL. If omitted, the value from
        :func:`get_database_url` is used.

    Returns
    -------
    sqlalchemy.engine.Engine
        A SQLAlchemy engine instance with ``future=True`` semantics.
    """
    if db_url is None:
        db_url = get_database_url()

    engine = create_engine(db_url, future=True)
    return engine
