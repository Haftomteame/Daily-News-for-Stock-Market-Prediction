"""Assemblage des données pour le treemap Market Carpet."""

from __future__ import annotations

import pandas as pd

from src.marketcarpet.loader import load_ohlcv_panel
from src.marketcarpet.metrics import compute_symbol_metrics
from src.marketcarpet.universe import load_universe


def build_market_carpet_df(
    group_key: str = "sp500",
    *,
    measurement: str = "performance",
    period_key: str = "1D",
    size_mode: str = "market_cap",
) -> tuple[pd.DataFrame, dict]:
    """
    Construit le DataFrame treemap : sector → symbol, avec métrique et taille.

    Retourne (df, meta) où meta contient des stats pour l'UI.
    """
    universe = load_universe(group_key)
    symbols = universe["symbol"].tolist()
    panel = load_ohlcv_panel(symbols)
    load_error = getattr(load_ohlcv_panel, "last_error", None)

    meta = {
        "group_key": group_key,
        "symbols_total": len(symbols),
        "symbols_with_data": 0,
        "data_from": None,
        "data_to": None,
        "pg_available": not panel.empty,
        "load_error": load_error,
    }

    if panel.empty:
        df = universe.copy()
        df["metric_value"] = None
        df["last_close"] = None
        df["size_value"] = df["market_cap_b"]
        df["tile_text"] = df["symbol"]
        df["color_label"] = "—"
        return df, meta

    metrics = compute_symbol_metrics(
        panel,
        measurement=measurement,
        period_key=period_key,
    )
    meta["symbols_with_data"] = len(metrics)
    meta["data_from"] = panel["date"].min()
    meta["data_to"] = panel["date"].max()

    df = universe.merge(metrics, on="symbol", how="left")

    if size_mode == "equal":
        df["size_value"] = 1.0
    elif size_mode == "price":
        df["size_value"] = df["last_close"].fillna(1.0).clip(lower=1.0)
    else:
        df["size_value"] = df["market_cap_b"].fillna(1.0).clip(lower=0.1)

    df = df.dropna(subset=["metric_value", "size_value"])

    suffix = "%" if measurement == "performance" else ""
    df["tile_text"] = df.apply(
        lambda r: (
            f"{r['symbol']}<br>{r['metric_value']:+.2f}{suffix}"
            if pd.notna(r["metric_value"])
            else str(r["symbol"])
        ),
        axis=1,
    )
    df["color_label"] = df["metric_value"].map(
        lambda v: f"{v:+.2f}{suffix}" if pd.notna(v) else "—"
    )
    return df.reset_index(drop=True), meta
