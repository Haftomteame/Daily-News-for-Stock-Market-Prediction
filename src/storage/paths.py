"""Chemins logiques lakehouse (local ou HDFS)."""

from __future__ import annotations

from src.storage.io import join_path, lakehouse_parquet, ml_file, monitoring_path


def bronze_parquet(table: str) -> str:
    return lakehouse_parquet("bronze", table)


def silver_parquet(table: str) -> str:
    return lakehouse_parquet("silver", table)


def gold_parquet() -> str:
    return lakehouse_parquet("gold", "daily_market_kpis")


def massive_cache_dir() -> str:
    return join_path("lakehouse", "bronze", "massive", "day_aggs")


def ml_model_path() -> str:
    return ml_file("market_direction_model.joblib")


def ml_metrics_path() -> str:
    return ml_file("metrics.json")


def ml_predictions_path() -> str:
    return ml_file("predictions.parquet")


def monitoring_history_path() -> str:
    return monitoring_path("metrics_history.parquet")


def monitoring_report_path(batch_id: str) -> str:
    return monitoring_path(f"report_{batch_id[:8]}.json")


def layer_table_dir(layer: str, table: str) -> str:
    return join_path("lakehouse", layer, table)
