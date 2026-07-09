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


def ensure_schema(name: str, *, eng: Engine | None = None) -> None:
    eng = eng or engine()
    with eng.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}"'))


def write_replace(table: str, df: pd.DataFrame, *, eng: Engine | None = None) -> None:
    """Remplace completement une table (DROP/CREATE implicite via pandas)."""
    eng = eng or engine()
    ensure_schema_public(eng)
    df.to_sql(table, eng, if_exists="replace", index=False, method="multi", chunksize=5000)


def read_table(table: str, *, eng: Engine | None = None) -> pd.DataFrame:
    eng = eng or engine()
    conn = eng.raw_connection()
    try:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()


def read_sql(sql: str, *, eng: Engine | None = None, params: dict | None = None) -> pd.DataFrame:
    eng = eng or engine()
    with eng.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def read_schema_table(
    schema: str,
    table: str,
    *,
    eng: Engine | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    eng = eng or engine()
    sql = f'SELECT * FROM "{schema}"."{table}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return read_sql(sql, eng=eng)


def list_schema_tables(schema: str, *, eng: Engine | None = None) -> list[str]:
    eng = eng or engine()
    df = read_sql(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        ORDER BY table_name
        """,
        eng=eng,
        params={"schema": schema},
    )
    return df["table_name"].tolist()


def schema_table_counts(schema: str, *, eng: Engine | None = None) -> dict[str, int]:
    """Compte les lignes par table d'un schema (requete legere via stats PG)."""
    eng = eng or engine()
    df = read_sql(
        """
        SELECT relname AS table_name, reltuples::bigint AS row_estimate
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relkind = 'r'
        ORDER BY relname
        """,
        eng=eng,
        params={"schema": schema},
    )
    return dict(zip(df["table_name"], df["row_estimate"], strict=False))


DOLT_WAREHOUSE_SCHEMAS = ("stocks", "options", "rates", "earnings")

DOLT_SCHEMA_LABELS = {
    "stocks": "Marché boursier",
    "options": "Options",
    "rates": "Taux d'intérêt",
    "earnings": "Résultats d'entreprises",
}

DOLT_SCHEMA_DESCRIPTIONS = {
    "stocks": "Cours, symboles, dividendes",
    "options": "Contrats et chaînes d'options",
    "rates": "Taux du Trésor américain",
    "earnings": "Bilans, BPA et estimations",
}

DOLT_TABLE_LABELS = {
    "ohlcv": "Cours boursiers (ouverture, clôture, volume…)",
    "symbol": "Liste des symboles",
    "dividend": "Dividendes versés",
    "split": "Fractionnements d'actions",
    "option_chain": "Chaînes d'options",
    "us_treasury": "Taux du Trésor US",
    "balance_sheet_assets": "Bilan — actifs",
    "balance_sheet_equity": "Bilan — capitaux propres",
    "balance_sheet_liabilities": "Bilan — dettes et passif",
    "cash_flow_statement": "Flux de trésorerie",
    "earnings_calendar": "Calendrier des publications",
    "eps_estimate": "Estimations de bénéfice par action",
    "eps_history": "Historique BPA",
    "income_statement": "Compte de résultat",
    "rank_score": "Scores de qualité financière",
    "sales_estimate": "Estimations de chiffre d'affaires",
}


def dolt_table_label(table_name: str) -> str:
    return DOLT_TABLE_LABELS.get(table_name, table_name.replace("_", " ").capitalize())


def warehouse_table_stats(
    schemas: tuple[str, ...] = DOLT_WAREHOUSE_SCHEMAS,
    *,
    eng: Engine | None = None,
) -> pd.DataFrame:
    """Lignes par table pour les schemas Dolt (stats PostgreSQL, rapide)."""
    eng = eng or engine()
    placeholders = ", ".join(f"'{s}'" for s in schemas)
    return read_sql(
        f"""
        SELECT schemaname AS schema, relname AS table_name, n_live_tup::bigint AS rows
        FROM pg_stat_user_tables
        WHERE schemaname IN ({placeholders})
        ORDER BY schemaname, relname
        """,
        eng=eng,
    )


def test_connection(*, eng: Engine | None = None) -> None:
    eng = eng or engine()
    with eng.begin() as conn:
        conn.execute(text("SELECT 1"))

