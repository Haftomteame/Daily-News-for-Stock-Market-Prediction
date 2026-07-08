"""Ecriture/lecture PostgreSQL (warehouse) pour tables structurees."""

from __future__ import annotations

import os
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def pg_dsn() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "wherehouse")
    user = os.getenv("PGUSER", "datax")
    password = os.getenv("PGPASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def pg_enabled() -> bool:
    # Active si un host est fourni (docker: postgres) et un password.
    return bool(os.getenv("PGHOST") and os.getenv("PGPASSWORD"))


def engine() -> Engine:
    return create_engine(pg_dsn(), pool_pre_ping=True)


def ensure_schema_public(eng: Engine) -> None:
    with eng.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))


def write_replace(table: str, df: pd.DataFrame, *, eng: Engine | None = None) -> None:
    """Remplace completement une table (DROP/CREATE implicite via pandas)."""
    eng = eng or engine()
    ensure_schema_public(eng)
    df.to_sql(table, eng, if_exists="replace", index=False, method="multi", chunksize=5000)


def read_table(table: str, *, eng: Engine | None = None) -> pd.DataFrame:
    eng = eng or engine()
    with eng.connect() as conn:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)


def test_connection(*, eng: Engine | None = None) -> None:
    eng = eng or engine()
    with eng.begin() as conn:
        conn.execute(text("SELECT 1"))

