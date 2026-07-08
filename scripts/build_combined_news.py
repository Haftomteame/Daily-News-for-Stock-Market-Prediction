"""Construit news_combined dans lakehouse/bronze/ depuis stock + reddit."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bronze.build_combined import build_combined_from_frames, stock_for_pipeline  # noqa: E402
from src.bronze.storage import bronze_exists, read_bronze_data, write_bronze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construit Combined_News dans lakehouse/bronze/news_combined/"
    )
    parser.add_argument(
        "--export-stock",
        type=Path,
        default=None,
        help="Optionnel: export CSV legacy upload_DJIA (debug uniquement)",
    )
    args = parser.parse_args()

    if not bronze_exists("stock_prices") or not bronze_exists("news_reddit"):
        print(
            "Erreur : bronze stock_prices et news_reddit requis.\n"
            "  python scripts/migrate_to_lakehouse.py\n"
            "  ou python scripts/fetch_massive_rest.py / fetch_reddit_news.py",
            file=sys.stderr,
        )
        sys.exit(1)

    stock = read_bronze_data("stock_prices")
    reddit = read_bronze_data("news_reddit")
    combined, labeled = build_combined_from_frames(stock, reddit)

    if combined.empty:
        print("Erreur : aucune date commune entre prix et Reddit.", file=sys.stderr)
        sys.exit(1)

    batch_id = str(uuid.uuid4())
    _, path = write_bronze(combined, "news_combined", batch_id, "build_combined")

    label_counts = combined["Label"].value_counts().to_dict()
    print(f"OK Combined : {len(combined):,} jours -> {path}")
    print(f"  Periode : {combined['Date'].min().date()} -> {combined['Date'].max().date()}")
    print(f"  Labels  : 0={label_counts.get(0, 0)} | 1={label_counts.get(1, 0)}")

    if args.export_stock:
        stock_out = stock_for_pipeline(labeled)
        args.export_stock.parent.mkdir(parents=True, exist_ok=True)
        stock_out.to_csv(args.export_stock, index=False)
        print(f"OK Stocks export debug : {args.export_stock}")

    print("Lancez : python pipeline/run_pipeline.py --massive --predict-year 2026")


if __name__ == "__main__":
    main()
