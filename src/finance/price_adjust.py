"""Ajustement des prix OHLCV pour fractionnements d'actions (splits)."""

from __future__ import annotations

import pandas as pd

from src.db.postgres import pg_enabled, read_sql

# Certains splits historiques sont etiquetes sous une classe (ex. GOOG vs GOOGL).
_SPLIT_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "GOOGL": ("GOOG",),
    "GOOG": ("GOOGL",),
}


def split_symbols_for_lookup(symbol: str) -> list[str]:
    sym = symbol.upper()
    return list(dict.fromkeys([sym, *_SPLIT_SYMBOL_ALIASES.get(sym, ())]))


def split_price_ratio(to_factor: float, for_factor: float) -> float:
    if not for_factor:
        return 1.0
    return float(to_factor) / float(for_factor)


def dedupe_splits(splits: pd.DataFrame) -> pd.DataFrame:
    """Elimine les annonces multiples d'un meme fractionnement (meme ratio)."""
    if splits.empty:
        return splits

    df = splits.copy()
    df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
    df["to_factor"] = pd.to_numeric(df["to_factor"], errors="coerce")
    df["for_factor"] = pd.to_numeric(df["for_factor"], errors="coerce")
    df = df.dropna(subset=["ex_date", "to_factor", "for_factor"])
    df["ratio"] = df.apply(
        lambda row: split_price_ratio(row["to_factor"], row["for_factor"]),
        axis=1,
    )
    df = df[df["ratio"] != 1.0]
    if df.empty:
        return df

    return (
        df.sort_values("ex_date")
        .groupby("ratio", as_index=False)
        .tail(1)
        .sort_values("ex_date")
        .reset_index(drop=True)
    )


def load_symbol_splits(symbol: str) -> pd.DataFrame:
    """Charge les fractionnements depuis PostgreSQL (symbole + alias connus)."""
    columns = ["act_symbol", "ex_date", "to_factor", "for_factor"]
    if not pg_enabled() or not symbol:
        return pd.DataFrame(columns=columns)

    symbols = split_symbols_for_lookup(symbol)
    placeholders = ", ".join(f":s{i}" for i in range(len(symbols)))
    params = {f"s{i}": sym for i, sym in enumerate(symbols)}
    try:
        df = read_sql(
            f"""
            SELECT act_symbol, ex_date, to_factor, for_factor
            FROM stocks.split
            WHERE act_symbol IN ({placeholders})
            ORDER BY ex_date
            """,
            params=params,
        )
    except Exception:
        return pd.DataFrame(columns=columns)
    return dedupe_splits(df)


def _effective_split_date(
    ex_date: pd.Timestamp,
    ratio: float,
    ohlcv: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.Timestamp:
    """Infere la premiere seance post-split a partir du saut de cours."""
    ex_date = pd.Timestamp(ex_date).normalize()
    if ohlcv.empty or "close" not in ohlcv.columns:
        return ex_date

    df = ohlcv.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    closes = pd.to_numeric(df.set_index(date_col)["close"], errors="coerce").dropna()
    if closes.empty:
        return ex_date

    session_dates = closes.index
    if ex_date not in session_dates:
        later = session_dates[session_dates >= ex_date]
        if len(later) == 0:
            return ex_date
        ex_date = later[0]

    ex_close = float(closes.loc[ex_date])
    if ex_close <= 0:
        return ex_date

    later_dates = session_dates[session_dates > ex_date]
    if len(later_dates) > 0:
        next_close = float(closes.loc[later_dates[0]])
        if next_close > 0 and abs(ex_close / next_close - ratio) / ratio < 0.08:
            return later_dates[0]

    earlier_dates = session_dates[session_dates < ex_date]
    if len(earlier_dates) > 0:
        prev_close = float(closes.loc[earlier_dates[-1]])
        if prev_close > 0 and abs(prev_close / ex_close - ratio) / ratio < 0.08:
            return ex_date

    return ex_date


def prepare_split_events(
    splits: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Convertit les lignes split en evenements avec date effective de cotation."""
    splits = dedupe_splits(splits)
    if splits.empty:
        return pd.DataFrame(columns=["effective_date", "ratio"])

    rows: list[dict] = []
    for _, row in splits.iterrows():
        rows.append({
            "effective_date": _effective_split_date(
                row["ex_date"],
                float(row["ratio"]),
                ohlcv,
                date_col=date_col,
            ),
            "ratio": float(row["ratio"]),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("effective_date")
        .reset_index(drop=True)
    )


def cumulative_split_factors(
    dates: pd.Series | pd.DatetimeIndex,
    splits: pd.DataFrame,
    *,
    ohlcv: pd.DataFrame | None = None,
    date_col: str = "date",
) -> pd.Series:
    """
    Facteur a diviser sur les prix bruts pour chaque date.

    Les cours a la date effective du split et apres restent inchanges ;
    l'historique anterieur est ramene au niveau post-split.
    """
    date_series = pd.Series(pd.to_datetime(dates, errors="coerce"))
    if splits.empty:
        return pd.Series(1.0, index=date_series.index)

    events = prepare_split_events(splits, ohlcv if ohlcv is not None else pd.DataFrame(), date_col=date_col)
    if events.empty:
        return pd.Series(1.0, index=date_series.index)

    split_dates = events["effective_date"].tolist()
    split_ratios = events["ratio"].tolist()

    factors: list[float] = []
    for trade_date in date_series:
        factor = 1.0
        if pd.isna(trade_date):
            factors.append(factor)
            continue
        trade_date = pd.Timestamp(trade_date).normalize()
        for effective_date, ratio in zip(split_dates, split_ratios, strict=False):
            if trade_date < effective_date:
                factor *= ratio
        factors.append(factor)
    return pd.Series(factors, index=date_series.index)


def adjust_ohlcv_for_splits(
    ohlcv: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    date_col: str = "date",
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Retourne une copie OHLCV avec prix et volumes ajustes pour les splits."""
    if ohlcv.empty or splits.empty:
        return ohlcv

    out = ohlcv.copy()
    factors = cumulative_split_factors(out[date_col], splits, ohlcv=out, date_col=date_col)
    if (factors == 1.0).all():
        return out

    for col in price_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / factors

    if volume_col in out.columns:
        out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce") * factors

    return out
