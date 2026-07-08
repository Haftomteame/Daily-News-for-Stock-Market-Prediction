"""Couche GOLD — KPIs agreges avec schema fixe."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.config import GOLD_SCHEMA, gold_data_path, silver_data_path
from src.storage.io import exists, query_duckdb, write_parquet


def build_daily_kpis(batch_id: str) -> tuple[pd.DataFrame, str]:
    """Agrege stock + news en KPIs journaliers (schema fixe)."""
    stock_path = silver_data_path("stock_prices")
    news_path = silver_data_path("news_reddit")

    if not exists(stock_path) or not exists(news_path):
        raise FileNotFoundError("Couches Silver requises avant Gold.")

    df = query_duckdb(
        """
        WITH stock AS (
            SELECT
                CAST(Date AS DATE) AS date,
                Open AS open,
                High AS high,
                Low AS low,
                Close AS close,
                CAST(Volume AS BIGINT) AS volume,
                (Close - LAG(Close) OVER (ORDER BY Date)) / NULLIF(LAG(Close) OVER (ORDER BY Date), 0) * 100
                    AS daily_return_pct,
                STDDEV(Close) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                    AS volatility_5d
            FROM silver_stock
            WHERE _quality_score = 1
        ),
        news_agg AS (
            SELECT
                CAST(Date AS DATE) AS date,
                COUNT(*) AS news_count,
                AVG(_headline_length) AS avg_headline_length,
                AVG(CASE WHEN _has_finance_keyword THEN 1.0 ELSE 0.0 END) AS finance_news_ratio
            FROM silver_news
            WHERE _quality_score = 1
            GROUP BY CAST(Date AS DATE)
        )
        SELECT
            s.date,
            s.open,
            s.high,
            s.low,
            s.close,
            s.volume,
            ROUND(s.daily_return_pct, 4) AS daily_return_pct,
            ROUND(s.volatility_5d, 4) AS volatility_5d,
            COALESCE(n.news_count, 0) AS news_count,
            ROUND(COALESCE(n.avg_headline_length, 0), 2) AS avg_headline_length,
            ROUND(COALESCE(n.finance_news_ratio, 0), 4) AS finance_news_ratio,
            CASE
                WHEN s.daily_return_pct > 0.1 THEN 'UP'
                WHEN s.daily_return_pct < -0.1 THEN 'DOWN'
                ELSE 'FLAT'
            END AS market_direction
        FROM stock s
        LEFT JOIN news_agg n ON s.date = n.date
        ORDER BY s.date
        """,
        {"silver_stock": stock_path, "silver_news": news_path},
    )

    df["computed_at"] = datetime.now(timezone.utc).isoformat()
    df["_batch_id"] = batch_id
    df["_layer"] = "gold"

    expected_cols = [col for col, _ in GOLD_SCHEMA] + ["_batch_id", "_layer"]
    df = df[expected_cols]

    path = gold_data_path()
    write_parquet(df, path)
    return df, path


def run_gold_aggregate(batch_id: str) -> dict:
    df, path = build_daily_kpis(batch_id)
    return {
        "batch_id": batch_id,
        "table": "daily_market_kpis",
        "rows": len(df),
        "path": path,
        "schema": GOLD_SCHEMA,
    }
