"""Client S3 Massive — Day Aggregates (flat files)."""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from src.config import MASSIVE_CACHE_DIR, MASSIVE_S3_BUCKET, MASSIVE_S3_ENDPOINT, MASSIVE_TICKER

S3_PREFIX = "us_stocks_sip/day_aggs_v1"


class MassiveDownloadError(Exception):
    """Erreur de telechargement Massive (credentials, plan, reseau)."""


def get_s3_client():
    access_key = os.getenv("MASSIVE_S3_ACCESS_KEY")
    secret_key = os.getenv("MASSIVE_S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise MassiveDownloadError(
            "MASSIVE_S3_ACCESS_KEY et MASSIVE_S3_SECRET_KEY requis "
            "(Dashboard Massive > Keys > Flat Files S3)."
        )
    return boto3.client(
        "s3",
        endpoint_url=MASSIVE_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def has_massive_credentials() -> bool:
    return bool(os.getenv("MASSIVE_S3_ACCESS_KEY") and os.getenv("MASSIVE_S3_SECRET_KEY"))


def s3_key(trade_date: date) -> str:
    return f"{S3_PREFIX}/{trade_date:%Y/%m/%Y-%m-%d}.csv.gz"


def list_available(client, year: int, month: int | None = None) -> list[str]:
    prefix = f"{S3_PREFIX}/{year}/"
    if month is not None:
        prefix = f"{S3_PREFIX}/{year}/{month:02d}/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MASSIVE_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return sorted(keys)


def download_day(client, trade_date: date, output_dir: str | os.PathLike[str]) -> str | None:
    key = s3_key(trade_date)
    from src.storage.io import exists as storage_exists, makedirs, write_bytes

    output_dir = os.fspath(output_dir)
    makedirs(output_dir)
    out_path = f"{output_dir.rstrip('/')}/{trade_date:%Y-%m-%d}.csv.gz"

    if storage_exists(out_path):
        return out_path

    try:
        obj = client.get_object(Bucket=MASSIVE_S3_BUCKET, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "403":
            raise MassiveDownloadError(
                "Acces S3 refuse (403). Day Aggregates requiert Stocks Starter+."
            ) from exc
        if code in {"404", "NoSuchKey"}:
            return None
        raise MassiveDownloadError(str(exc)) from exc

    write_bytes(obj["Body"].read(), out_path)
    return out_path


def _parse_day_file(path: str, ticker: str) -> dict | None:
    from src.storage.io import read_bytes

    df = pd.read_csv(gzip.open(io.BytesIO(read_bytes(path))))
    row = df[df["ticker"] == ticker.upper()]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "Date": pd.to_datetime(int(r["window_start"]), unit="ns"),
        "Open": float(r["open"]),
        "High": float(r["high"]),
        "Low": float(r["low"]),
        "Close": float(r["close"]),
        "Volume": int(r["volume"]),
        "Adj Close": float(r["close"]),
    }


def download_range(
    start: date,
    end: date,
    cache_dir: str | None = None,
    client=None,
) -> list[str]:
    cache_dir = cache_dir or MASSIVE_CACHE_DIR
    client = client or get_s3_client()
    downloaded: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            path = download_day(client, current, cache_dir)
            if path:
                downloaded.append(path)
        current += timedelta(days=1)
    return downloaded


def build_stock_prices_from_cache(
    ticker: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame DJIA-compatible depuis les fichiers .csv.gz en cache."""
    from src.storage.io import glob_paths

    cache_dir = cache_dir or MASSIVE_CACHE_DIR
    ticker = (ticker or MASSIVE_TICKER).upper()
    files = glob_paths(f"{cache_dir.rstrip('/')}/*.csv.gz")
    if not files:
        raise FileNotFoundError(f"Aucun fichier Massive dans {cache_dir}")

    rows: list[dict] = []
    for path in files:
        row = _parse_day_file(path, ticker)
        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"Aucune donnee pour le ticker {ticker} dans {cache_dir}")

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def fetch_stock_prices(
    start: date,
    end: date,
    ticker: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Telecharge (si besoin) puis assemble les prix journaliers d'un ticker."""
    cache_dir = cache_dir or MASSIVE_CACHE_DIR
    ticker = ticker or MASSIVE_TICKER
    client = get_s3_client()
    download_range(start, end, cache_dir, client)
    return build_stock_prices_from_cache(ticker, cache_dir)


def infer_date_range_from_news() -> tuple[date, date]:
    """Plage de dates alignee sur les actualites Reddit (bronze lakehouse)."""
    from src.bronze.storage import bronze_exists, read_bronze_data

    if bronze_exists("news_reddit"):
        news = read_bronze_data("news_reddit")
    else:
        from src.config import LEGACY_DATA_DIR

        legacy_candidates = [
            LEGACY_DATA_DIR / "RedditNews_2024_2026.csv",
            LEGACY_DATA_DIR / "RedditNews.csv",
        ]
        legacy = next((p for p in legacy_candidates if p.exists()), None)
        if legacy is None:
            raise FileNotFoundError(
                "Bronze news_reddit absent. Executez --refresh-bronze ou migrate_to_lakehouse.py"
            )
        news = pd.read_csv(legacy)

    dates = pd.to_datetime(news["Date"], errors="coerce").dropna()
    return dates.min().date(), dates.max().date()
