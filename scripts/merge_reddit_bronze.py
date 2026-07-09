#!/usr/bin/env python3
"""Fusionne les tables Bronze news_reddit_* en news_reddit."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.bronze.storage import read_bronze_data, write_bronze  # noqa: E402


def _discover_tables() -> list[str]:
    bronze_root = PROJECT_ROOT / "lakehouse" / "bronze"
    tables = []
    if not bronze_root.is_dir():
        return tables
    for path in bronze_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name == "news_reddit" or name.startswith("news_reddit_"):
            if (path / "data.parquet").exists():
                tables.append(name)
    return sorted(set(tables))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusionne news_reddit_* -> news_reddit")
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Tables a fusionner (defaut: toutes news_reddit*)",
    )
    parser.add_argument(
        "--target",
        default="news_reddit",
        help="Table Bronze cible (defaut: news_reddit)",
    )
    args = parser.parse_args()

    tables = args.tables or _discover_tables()
    if not tables:
        print("Aucune table news_reddit* trouvee.", file=sys.stderr)
        sys.exit(1)

    frames = [read_bronze_data(t) for t in tables]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "News"]).sort_values(["Date", "News"])

    _, path = write_bronze(df, args.target, str(uuid.uuid4()), "merge_reddit_bronze")
    print(f"OK {len(df):,} lignes depuis {tables} -> {path}")
    print(f"Periode : {df['Date'].min()} -> {df['Date'].max()}")


if __name__ == "__main__":
    main()
