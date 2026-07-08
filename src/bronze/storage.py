"""Lecture / ecriture couche Bronze (source de verite lakehouse)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.storage.paths import bronze_parquet
from src.storage.io import exists, read_parquet, write_parquet

BRONZE_METADATA_COLS = ("_ingestion_ts", "_source_file", "_batch_id", "_layer")


def bronze_data_path(table: str) -> str:
    return bronze_parquet(table)


def bronze_exists(table: str) -> bool:
    return exists(bronze_parquet(table))


def bronze_metadata(source_label: str, batch_id: str) -> dict:
    return {
        "_ingestion_ts": datetime.now(timezone.utc).isoformat(),
        "_source_file": source_label,
        "_batch_id": batch_id,
        "_layer": "bronze",
    }


def strip_metadata(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in BRONZE_METADATA_COLS if col in df.columns]
    return df.drop(columns=drop_cols)


def read_bronze_raw(table: str) -> pd.DataFrame:
    path = bronze_parquet(table)
    if not exists(path):
        raise FileNotFoundError(f"Bronze absent : {path}")
    return read_parquet(path)


def read_bronze_data(table: str) -> pd.DataFrame:
    return strip_metadata(read_bronze_raw(table))


def write_bronze(
    df: pd.DataFrame,
    table: str,
    batch_id: str,
    source_label: str,
) -> tuple[pd.DataFrame, str]:
    path = bronze_parquet(table)
    payload = strip_metadata(df).copy()
    meta = bronze_metadata(source_label, batch_id)
    for key, value in meta.items():
        payload[key] = value
    write_parquet(payload, path)
    return payload, path


def append_bronze_rows(
    rows: list[dict],
    table: str,
    batch_id: str,
    source_label: str,
    *,
    dedupe_on: str = "Date",
) -> str:
    """Ajoute des lignes a une table Bronze (merge + dedupe, metadata mises a jour)."""
    if not rows:
        path = bronze_parquet(table)
        return path

    incoming = pd.DataFrame(rows)
    if dedupe_on in incoming.columns:
        incoming[dedupe_on] = pd.to_datetime(incoming[dedupe_on])

    path = bronze_parquet(table)
    if exists(path):
        existing = strip_metadata(read_parquet(path))
        if dedupe_on in existing.columns:
            existing[dedupe_on] = pd.to_datetime(existing[dedupe_on])
        combined = pd.concat([existing, incoming], ignore_index=True)
        if dedupe_on in combined.columns:
            combined = combined.drop_duplicates(subset=[dedupe_on], keep="last")
        combined = combined.sort_values(dedupe_on).reset_index(drop=True)
    else:
        combined = incoming.sort_values(dedupe_on).reset_index(drop=True)

    write_bronze(combined, table, batch_id, source_label)
    return path
