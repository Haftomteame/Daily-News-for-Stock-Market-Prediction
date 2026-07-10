"""Ecriture/lecture PostgreSQL (warehouse) pour tables structurees."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def pg_host() -> str:
    """Resout PGHOST : `postgres` (Docker) -> `localhost` hors conteneur."""
    host = os.getenv("PGHOST", "localhost")
    if host != "postgres":
        return host
    if Path("/.dockerenv").exists():
        return host
    try:
        socket.getaddrinfo(host, None)
        return host
    except socket.gaierror:
        return "localhost"


def pg_connection_hint() -> str:
    configured = os.getenv("PGHOST", "localhost")
    resolved = pg_host()
    if configured == "postgres" and resolved == "localhost":
        return (
            "`PGHOST=postgres` ne fonctionne qu'à l'intérieur de Docker ; "
            "connexion automatique via `localhost`. "
            "Vérifiez que PostgreSQL tourne (`docker compose --profile hdfs up -d postgres`)."
        )
    return (
        f"Vérifiez que PostgreSQL écoute sur `{resolved}` "
        f"(port {os.getenv('PGPORT', '5432')}) et que le mot de passe est correct."
    )


def pg_dsn() -> str:
    host = pg_host()
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
    "stocks": "Actions",
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
    "ohlcv": "Prix des actions (ouverture, clôture, volume…)",
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


# Index pour les filtres act_symbol du dashboard (COUNT / dernière date).
WAREHOUSE_INDEXES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("options", "option_chain", "idx_option_chain_act_symbol_date", ("act_symbol", "date")),
    ("stocks", "ohlcv", "idx_ohlcv_act_symbol_date", ("act_symbol", "date")),
    ("stocks", "dividend", "idx_dividend_act_symbol", ("act_symbol",)),
    ("stocks", "split", "idx_split_act_symbol", ("act_symbol",)),
    ("earnings", "earnings_calendar", "idx_earnings_calendar_act_symbol", ("act_symbol",)),
    ("earnings", "eps_estimate", "idx_eps_estimate_act_symbol", ("act_symbol",)),
    ("earnings", "eps_history", "idx_eps_history_act_symbol", ("act_symbol",)),
    ("earnings", "balance_sheet_assets", "idx_balance_sheet_assets_act_symbol", ("act_symbol",)),
    ("earnings", "balance_sheet_equity", "idx_balance_sheet_equity_act_symbol", ("act_symbol",)),
    ("earnings", "balance_sheet_liabilities", "idx_balance_sheet_liabilities_act_symbol", ("act_symbol",)),
    ("earnings", "cash_flow_statement", "idx_cash_flow_statement_act_symbol", ("act_symbol",)),
    ("earnings", "income_statement", "idx_income_statement_act_symbol", ("act_symbol",)),
    ("earnings", "rank_score", "idx_rank_score_act_symbol", ("act_symbol",)),
    ("earnings", "sales_estimate", "idx_sales_estimate_act_symbol", ("act_symbol",)),
)


def _index_exists(schema: str, index_name: str, *, eng: Engine) -> bool:
    df = read_sql(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = :schema AND indexname = :index_name
        LIMIT 1
        """,
        eng=eng,
        params={"schema": schema, "index_name": index_name},
    )
    return not df.empty


def _table_exists(schema: str, table: str, *, eng: Engine) -> bool:
    df = read_sql(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_name = :table
        LIMIT 1
        """,
        eng=eng,
        params={"schema": schema, "table": table},
    )
    return not df.empty


def ensure_warehouse_indexes(*, eng: Engine | None = None) -> list[str]:
    """Crée les index manquants pour accélérer les requêtes filtrées par symbole."""
    eng = eng or engine()
    created: list[str] = []
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for schema, table, index_name, columns in WAREHOUSE_INDEXES:
            if _index_exists(schema, index_name, eng=eng):
                continue
            if not _table_exists(schema, table, eng=eng):
                continue
            cols = ", ".join(columns)
            conn.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                    f'ON "{schema}"."{table}" ({cols})'
                )
            )
            created.append(f"{schema}.{index_name}")
    return created


def option_chain_latest_count(symbol: str, *, eng: Engine | None = None) -> int:
    """Nombre de contrats à la dernière date disponible pour un symbole."""
    eng = eng or engine()
    df = read_sql(
        """
        SELECT COUNT(*) AS n
        FROM options.option_chain
        WHERE act_symbol = :symbol
          AND date = (
              SELECT date
              FROM options.option_chain
              WHERE act_symbol = :symbol
              ORDER BY date DESC
              LIMIT 1
          )
        """,
        eng=eng,
        params={"symbol": symbol.upper()},
    )
    return int(df.iloc[0]["n"] or 0)


def symbol_row_count(
    schema: str,
    table: str,
    symbol: str,
    *,
    eng: Engine | None = None,
) -> int:
    """Compte les lignes d'une table warehouse pour un symbole."""
    eng = eng or engine()
    symbol = symbol.upper()
    if table == "symbol":
        sql = "SELECT COUNT(*) AS n FROM stocks.symbol WHERE act_symbol = :symbol"
    elif schema == "options" and table == "option_chain":
        return option_chain_latest_count(symbol, eng=eng)
    else:
        sql = f'SELECT COUNT(*) AS n FROM "{schema}"."{table}" WHERE act_symbol = :symbol'
    df = read_sql(sql, eng=eng, params={"symbol": symbol})
    return int(df.iloc[0]["n"] or 0)

