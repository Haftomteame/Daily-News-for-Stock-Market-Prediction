"""Chargement OHLCV multi-symboles depuis PostgreSQL."""

from __future__ import annotations

import pandas as pd

from src.db.postgres import pg_enabled, read_sql
from src.finance.price_adjust import adjust_ohlcv_for_splits, load_symbol_splits


def _bind_in_clause(symbols: list[str]) -> tuple[str, dict]:
    params: dict = {}
    placeholders = []
    for i, sym in enumerate(symbols):
        key = f"s{i}"
        placeholders.append(f":{key}")
        params[key] = sym.upper()
    return ", ".join(placeholders), params


def load_ohlcv_panel(
    symbols: list[str],
    *,
    lookback_days: int = 400,
) -> pd.DataFrame:
    """Panel journalier (symbol, date, close, volume) pour une liste de symboles."""
    if not symbols or not pg_enabled():
        return pd.DataFrame(columns=["symbol", "date", "close", "volume"])

    in_clause, params = _bind_in_clause(symbols)
    params["lookback"] = int(lookback_days)
    sql = f"""
        SELECT act_symbol AS symbol, date, close, volume
        FROM stocks.ohlcv
        WHERE act_symbol IN ({in_clause})
          AND CAST(date AS DATE) >= CURRENT_DATE - CAST(:lookback AS INTEGER)
        ORDER BY act_symbol, date
    """
    try:
        df = read_sql(sql, params=params)
    except Exception as exc:
        load_ohlcv_panel.last_error = str(exc)  # type: ignore[attr-defined]
        return pd.DataFrame(columns=["symbol", "date", "close", "volume"])

    load_ohlcv_panel.last_error = None  # type: ignore[attr-defined]

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values(["symbol", "date"])

    adjusted_chunks: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=False):
        splits = load_symbol_splits(symbol)
        adjusted_chunks.append(adjust_ohlcv_for_splits(grp, splits))
    return pd.concat(adjusted_chunks, ignore_index=True)
