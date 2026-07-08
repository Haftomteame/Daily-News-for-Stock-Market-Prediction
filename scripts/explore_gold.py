#!/usr/bin/env python3
"""Exploration des KPIs Gold (schema fixe)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOLD_SCHEMA, gold_data_path
from src.env import load_dotenv
from src.storage.io import exists, query_duckdb, storage_label

load_dotenv()


def main() -> int:
    gold_path = gold_data_path()
    if not exists(gold_path):
        print("Couche Gold absente. Executez d'abord : python pipeline/run_pipeline.py")
        return 1

    print("=" * 60)
    print("  COUCHE GOLD - daily_market_kpis")
    print(f"  Stockage : {storage_label()}")
    print("=" * 60)

    print("\nSchema fixe :")
    for col, dtype in GOLD_SCHEMA:
        print(f"  - {col}: {dtype}")

    stats = query_duckdb(
        """
        SELECT
            COUNT(*) AS jours,
            MIN(date) AS date_min,
            MAX(date) AS date_max,
            ROUND(AVG(daily_return_pct), 4) AS rendement_moy_pct,
            ROUND(AVG(news_count), 1) AS news_moy_jour,
            ROUND(AVG(finance_news_ratio), 4) AS ratio_finance_moy
        FROM gold_table
        """,
        {"gold_table": gold_path},
    )
    print("\nStatistiques globales :")
    print(stats.to_string(index=False))

    sample = query_duckdb(
        """
        SELECT date, close, daily_return_pct, news_count, finance_news_ratio, market_direction
        FROM gold_table
        ORDER BY date DESC
        LIMIT 10
        """,
        {"gold_table": gold_path},
    )
    print("\n10 derniers jours :")
    print(sample.to_string(index=False))

    directions = query_duckdb(
        """
        SELECT market_direction, COUNT(*) AS nb
        FROM gold_table
        GROUP BY market_direction
        ORDER BY nb DESC
        """,
        {"gold_table": gold_path},
    )
    print("\nRepartition market_direction :")
    print(directions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
