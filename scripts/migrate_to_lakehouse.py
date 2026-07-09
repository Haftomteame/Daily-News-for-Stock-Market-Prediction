#!/usr/bin/env python3
"""Importe les CSV legacy (Data/) vers lakehouse/bronze/ (migration ponctuelle)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bronze.build_combined import build_combined_from_frames  # noqa: E402
from src.bronze.storage import write_bronze  # noqa: E402
from src.config import LEGACY_DATA_DIR  # noqa: E402

import pandas as pd  # noqa: E402


def _import_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} introuvable : {path}")
    return pd.read_csv(path, low_memory=False) if label == "combined" else pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migre Data/*.csv vers lakehouse/bronze/")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=LEGACY_DATA_DIR,
        help="Dossier CSV legacy (defaut: Data/)",
    )
    parser.add_argument(
        "--rebuild-combined",
        action="store_true",
        help="Reconstruit news_combined depuis stock + reddit au lieu d'importer le CSV",
    )
    args = parser.parse_args()

    batch_id = str(uuid.uuid4())
    data_dir = args.data_dir

    stock_candidates = [
        data_dir / "upload_DJIA_2024_2026.csv",
        data_dir / "massive_rest.csv",
        data_dir / "upload_DJIA_table.csv",
    ]
    reddit_candidates = [
        data_dir / "RedditNews_2024_2026.csv",
        data_dir / "RedditNews.csv",
    ]
    combined_candidates = [
        data_dir / "Combined_News_DJIA_2024_2026.csv",
        data_dir / "Combined_News_DJIA.csv",
    ]

    stock_path = next((p for p in stock_candidates if p.exists()), None)
    reddit_path = next((p for p in reddit_candidates if p.exists()), None)
    combined_path = next((p for p in combined_candidates if p.exists()), None)

    if not stock_path or not reddit_path:
        print("Erreur : CSV stock et/ou Reddit introuvables dans", data_dir, file=sys.stderr)
        return 1

    stock_df = _import_csv(stock_path, "stock")
    reddit_df = _import_csv(reddit_path, "reddit")

    _, stock_path_out = write_bronze(stock_df, "stock_prices", batch_id, stock_path.name)
    _, reddit_path_out = write_bronze(reddit_df, "news_reddit", batch_id, reddit_path.name)

    if args.rebuild_combined:
        combined_df, _ = build_combined_from_frames(stock_df, reddit_df)
        combined_label = "build_combined"
    elif combined_path:
        combined_df = _import_csv(combined_path, "combined")
        combined_label = combined_path.name
    else:
        combined_df, _ = build_combined_from_frames(stock_df, reddit_df)
        combined_label = "build_combined"

    _, combined_path_out = write_bronze(combined_df, "news_combined", batch_id, combined_label)

    print("OK Migration vers lakehouse/bronze/")
    print(f"  stock_prices   : {len(stock_df):,} lignes -> {stock_path_out}")
    print(f"  news_reddit    : {len(reddit_df):,} lignes -> {reddit_path_out}")
    print(f"  news_combined  : {len(combined_df):,} lignes -> {combined_path_out}")
    print("\nLancez : python pipeline/run_pipeline.py --massive --predict-year 2026")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
