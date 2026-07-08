"""Couche SILVER — nettoyage et enrichissement metadata."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from src.config import silver_data_path
from src.storage.io import exists, read_parquet, write_parquet

FINANCE_KEYWORDS = {
    "stock", "market", "dow", "djia", "nasdaq", "s&p", "trading", "investor",
    "economy", "fed", "inflation", "gdp", "earnings", "wall street", "shares",
    "bond", "interest rate", "recession", "bull", "bear", "ipo", "merger",
}


def _silver_metadata(batch_id: str) -> dict:
    return {
        "_processed_ts": datetime.now(timezone.utc).isoformat(),
        "_batch_id": batch_id,
        "_layer": "silver",
    }


def _write_parquet(df: pd.DataFrame, table: str) -> str:
    path = silver_data_path(table)
    write_parquet(df, path)
    return path


def _load_bronze(table: str) -> pd.DataFrame:
    from src.config import bronze_data_path

    path = bronze_data_path(table)
    if not exists(path):
        raise FileNotFoundError(f"Bronze non trouve : {path}")
    return read_parquet(path)


def transform_stock_prices(batch_id: str) -> tuple[pd.DataFrame, str]:
    """Nettoyage structure + flags qualite."""
    df = _load_bronze("stock_prices").copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["Date"], keep="first")
    df = df.sort_values("Date").reset_index(drop=True)

    df["_is_null_price"] = df["Close"].isna()
    df["_is_invalid_ohlc"] = (df["High"] < df["Low"]) | (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    df["_quality_score"] = (~df["_is_null_price"] & ~df["_is_invalid_ohlc"]).astype(float)

    meta = _silver_metadata(batch_id)
    for key, value in meta.items():
        df[key] = value

    path = _write_parquet(df, "stock_prices")
    return df, path


def _clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _has_finance_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in FINANCE_KEYWORDS)


def transform_news_reddit(batch_id: str) -> tuple[pd.DataFrame, str]:
    """Nettoyage non structure + metadata textuelle."""
    df = _load_bronze("news_reddit").copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["News"] = df["News"].apply(_clean_text)

    df = df[df["News"].str.len() > 0]
    df = df.drop_duplicates(subset=["Date", "News"], keep="first")

    df["_headline_length"] = df["News"].str.len()
    df["_word_count"] = df["News"].str.split().str.len()
    df["_has_finance_keyword"] = df["News"].apply(_has_finance_keyword)
    df["_is_empty"] = df["News"].str.len() == 0
    df["_quality_score"] = (~df["Date"].isna() & ~df["_is_empty"]).astype(float)

    meta = _silver_metadata(batch_id)
    for key, value in meta.items():
        df[key] = value

    path = _write_parquet(df, "news_reddit")
    return df, path


def _clean_combined_headline(text: str) -> str:
    if pd.isna(text):
        return ""
    cleaned = str(text).strip()
    if cleaned.startswith("b'") and cleaned.endswith("'"):
        cleaned = cleaned[2:-1]
    elif cleaned.startswith('b"') and cleaned.endswith('"'):
        cleaned = cleaned[2:-1]
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def transform_news_combined(batch_id: str) -> tuple[pd.DataFrame, str]:
    """Nettoyage hybride (news + label ML) + features textuelles."""
    df = _load_bronze("news_combined").copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

    top_cols = [c for c in df.columns if c.startswith("Top")]
    for col in top_cols:
        df[col] = df[col].apply(_clean_combined_headline)

    def _row_finance_ratio(row: pd.Series) -> float:
        texts = [row[c] for c in top_cols if row[c]]
        if not texts:
            return 0.0
        hits = sum(1 for t in texts if _has_finance_keyword(t))
        return hits / len(texts)

    def _row_avg_length(row: pd.Series) -> float:
        lengths = [len(row[c]) for c in top_cols if row[c]]
        return sum(lengths) / len(lengths) if lengths else 0.0

    df["_combined_finance_ratio"] = df.apply(_row_finance_ratio, axis=1)
    df["_combined_avg_length"] = df.apply(_row_avg_length, axis=1)
    df["_headline_count"] = df[top_cols].apply(lambda r: sum(1 for v in r if v), axis=1)
    df["_quality_score"] = (
        df["Date"].notna() & df["Label"].isin([0, 1]) & (df["_headline_count"] > 0)
    ).astype(float)

    meta = _silver_metadata(batch_id)
    for key, value in meta.items():
        df[key] = value

    path = _write_parquet(df, "news_combined")
    return df, path


def run_silver_transform(batch_id: str) -> dict:
    """Execute les transformations Silver."""
    results = {}
    for name, fn in [
        ("stock_prices", transform_stock_prices),
        ("news_reddit", transform_news_reddit),
        ("news_combined", transform_news_combined),
    ]:
        df, path = fn(batch_id)
        results[name] = {"rows": len(df), "path": path}
    return {"batch_id": batch_id, "tables": results}
