"""Construit Combined_News_DJIA depuis prix + RedditNews."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TOP_N = 25


def _normalize_dates(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col])


def load_stock_prices(source: Path | pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(source) if isinstance(source, Path) else source.copy()
    df = _normalize_dates(df)
    for col in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")


def compute_labels(stock: pd.DataFrame) -> pd.DataFrame:
    """Label Kaggle : 1 si Open du lendemain > Open du jour."""
    df = stock.sort_values("Date").copy()
    df["Label"] = (df["Open"].shift(-1) > df["Open"]).astype("Int64")
    df = df[df["Label"].notna()].copy()
    df["Label"] = df["Label"].astype(int)
    return df


def aggregate_reddit_tops(reddit: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    df = _normalize_dates(reddit)
    df["News"] = df["News"].astype(str).str.strip()
    df = df[df["News"].ne("") & df["News"].ne("nan")]

    rows: list[dict] = []
    for day, group in df.groupby("Date"):
        titles = group["News"].drop_duplicates().head(top_n).tolist()
        row: dict = {"Date": day}
        for i in range(1, top_n + 1):
            row[f"Top{i}"] = titles[i - 1] if i <= len(titles) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_from_frames(
    stock: pd.DataFrame,
    reddit: pd.DataFrame,
    top_n: int = TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (combined, stock_avec_labels) depuis DataFrames bronze."""
    stock_df = load_stock_prices(stock)
    labeled = compute_labels(stock_df)
    tops = aggregate_reddit_tops(_normalize_dates(reddit.copy()), top_n=top_n)

    combined = labeled[["Date", "Label"]].merge(tops, on="Date", how="inner")
    combined = combined.sort_values("Date").reset_index(drop=True)

    top_cols = [f"Top{i}" for i in range(1, top_n + 1)]
    combined = combined[["Date", "Label", *top_cols]]
    return combined, labeled
