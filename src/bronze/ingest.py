"""Couche BRONZE — ingestion ELT (donnees brutes dans lakehouse/)."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from src.bronze.build_combined import build_combined_from_frames
from src.bronze.storage import bronze_exists, read_bronze_data, write_bronze
from src.config import MASSIVE_CACHE_DIR, MASSIVE_TICKER
from src.storage.io import glob_paths


def _load_stock_prices_from_massive() -> pd.DataFrame:
    """Cache S3 -> API REST Massive."""
    from src.bronze.massive import (
        MassiveDownloadError,
        build_stock_prices_from_cache,
        fetch_stock_prices,
        has_massive_credentials,
        infer_date_range_from_news,
    )
    from src.bronze.massive_rest import MassiveRestError, fetch_daily_bars, has_massive_api_key

    start, end = infer_date_range_from_news()

    if glob_paths(f"{MASSIVE_CACHE_DIR.rstrip('/')}/*.csv.gz"):
        return build_stock_prices_from_cache(MASSIVE_TICKER, MASSIVE_CACHE_DIR)

    if has_massive_credentials():
        try:
            return fetch_stock_prices(start, end, MASSIVE_TICKER, MASSIVE_CACHE_DIR)
        except MassiveDownloadError:
            pass

    if has_massive_api_key():
        return fetch_daily_bars(MASSIVE_TICKER, start, end)

    raise FileNotFoundError(
        "Aucune source Massive : cache S3, telechargement S3 ou MASSIVE_API_KEY."
    )


def _fetch_news_reddit(start: date, end: date) -> pd.DataFrame:
    from src.bronze.reddit_fetch import RedditFetchError, fetch_range_arctic

    try:
        return fetch_range_arctic(start, end, subreddits=["worldnews"])
    except RedditFetchError as exc:
        raise FileNotFoundError(f"Fetch Reddit impossible : {exc}") from exc


def _infer_reddit_date_range(stock: pd.DataFrame | None = None) -> tuple[date, date]:
    if bronze_exists("news_reddit"):
        news = read_bronze_data("news_reddit")
        dates = pd.to_datetime(news["Date"], errors="coerce").dropna()
        return dates.min().date(), dates.max().date()

    if stock is not None and not stock.empty:
        dates = pd.to_datetime(stock["Date"], errors="coerce").dropna()
        return dates.min().date(), dates.max().date()

    from src.bronze.massive import infer_date_range_from_news

    return infer_date_range_from_news()


def ingest_stock_prices(
    batch_id: str | None = None,
    *,
    refresh: bool = False,
    source: str = "lakehouse",
) -> tuple[pd.DataFrame, str]:
    """Ingestion structuree : cours DJIA dans lakehouse/bronze/stock_prices/."""
    from src.bronze.massive_rest import MassiveRestError

    batch_id = batch_id or str(uuid.uuid4())

    if refresh or not bronze_exists("stock_prices"):
        if source in {"massive", "auto", "lakehouse"}:
            try:
                df = _load_stock_prices_from_massive()
                source_label = f"massive_{MASSIVE_TICKER.lower()}"
            except (FileNotFoundError, ValueError, MassiveRestError) as exc:
                if bronze_exists("stock_prices"):
                    df = read_bronze_data("stock_prices")
                    source_label = "lakehouse_cache"
                    print(f"  WARN Massive indisponible ({exc}), bronze existant conserve.")
                else:
                    raise FileNotFoundError(
                        "Bronze stock_prices absent. Lancez : "
                        "python pipeline/run_pipeline.py --refresh-bronze --massive"
                    ) from exc
        else:
            raise ValueError(f"source inconnue : {source}")
    else:
        df = read_bronze_data("stock_prices")
        source_label = "lakehouse"

    return write_bronze(df, "stock_prices", batch_id, source_label)


def ingest_news_reddit(
    batch_id: str | None = None,
    *,
    refresh: bool = False,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[pd.DataFrame, str]:
    """Ingestion non structuree : actualites Reddit dans lakehouse/bronze/news_reddit/."""
    batch_id = batch_id or str(uuid.uuid4())

    if refresh or not bronze_exists("news_reddit"):
        stock = read_bronze_data("stock_prices") if bronze_exists("stock_prices") else None
        start, end = date_from, date_to
        if start is None or end is None:
            start, end = _infer_reddit_date_range(stock)
        df = _fetch_news_reddit(start, end)
        source_label = "arctic_shift"
    else:
        df = read_bronze_data("news_reddit")
        source_label = "lakehouse"

    return write_bronze(df, "news_reddit", batch_id, source_label)


def ingest_news_combined(batch_id: str | None = None) -> tuple[pd.DataFrame, str]:
    """Ingestion hybride : construit Combined depuis bronze stock + reddit."""
    batch_id = batch_id or str(uuid.uuid4())

    if not bronze_exists("stock_prices") or not bronze_exists("news_reddit"):
        raise FileNotFoundError(
            "Bronze stock_prices et news_reddit requis. "
            "Executez --refresh-bronze ou scripts/migrate_to_lakehouse.py"
        )

    stock = read_bronze_data("stock_prices")
    reddit = read_bronze_data("news_reddit")
    combined, _ = build_combined_from_frames(stock, reddit)

    if combined.empty:
        raise ValueError("Aucune date commune entre prix et Reddit dans le bronze.")

    return write_bronze(combined, "news_combined", batch_id, "build_combined")


def ingest_massive_day_aggs(batch_id: str | None = None) -> tuple[pd.DataFrame, str]:
    """Ingestion brute : fichiers day aggregates Massive (multi-tickers)."""
    batch_id = batch_id or str(uuid.uuid4())
    from src.bronze.massive import fetch_stock_prices, has_massive_credentials, infer_date_range_from_news

    start, end = infer_date_range_from_news()
    if has_massive_credentials() and not glob_paths(f"{MASSIVE_CACHE_DIR.rstrip('/')}/*.csv.gz"):
        fetch_stock_prices(start, end, cache_dir=MASSIVE_CACHE_DIR)

    files = glob_paths(f"{MASSIVE_CACHE_DIR.rstrip('/')}/*.csv.gz")
    if not files:
        raise FileNotFoundError(
            "Aucun fichier Massive dans le cache lakehouse. "
            "Configurez MASSIVE_S3_ACCESS_KEY ou telechargez via scripts/fetch_massive_day_aggs.py"
        )

    import gzip
    import io
    from src.storage.io import read_bytes

    frames = [pd.read_csv(gzip.open(io.BytesIO(read_bytes(file_path)))) for file_path in files]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["window_start"], unit="ns")

    return write_bronze(df, "massive_day_aggs", batch_id, "massive_day_aggs")


def run_bronze_ingestion(
    batch_id: str | None = None,
    *,
    refresh: bool = False,
    stock_source: str = "lakehouse",
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict:
    """Execute l'ingestion ELT de toutes les sources vers lakehouse/bronze/."""
    batch_id = batch_id or str(uuid.uuid4())
    results = {}
    for name, fn, kwargs in [
        ("stock_prices", ingest_stock_prices, {"refresh": refresh, "source": stock_source}),
        ("news_reddit", ingest_news_reddit, {"refresh": refresh, "date_from": date_from, "date_to": date_to}),
        ("news_combined", ingest_news_combined, {}),
    ]:
        df, path = fn(batch_id, **kwargs)
        results[name] = {"rows": len(df), "path": str(path), "columns": list(df.columns)}
    return {"batch_id": batch_id, "tables": results}
