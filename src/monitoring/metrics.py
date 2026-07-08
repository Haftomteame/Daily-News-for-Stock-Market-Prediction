"""Monitoring — cout, latence et qualite par couche."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import DATA_TYPES, MONITORING_HISTORY_PATH
from src.storage.io import (
    dir_size,
    exists,
    query_duckdb,
    read_parquet,
    write_json,
    write_parquet,
)
from src.storage.paths import layer_table_dir, monitoring_report_path


def _quality_checks(layer: str, parquet_path: str) -> dict[str, Any]:
    if not exists(parquet_path):
        return {"error": "fichier absent", "quality_score": 0.0}

    df_info = query_duckdb(
        "SELECT COUNT(*) AS row_count FROM data_table",
        {"data_table": parquet_path},
    )
    row_count = int(df_info["row_count"].iloc[0])
    checks: dict[str, Any] = {"row_count": row_count}

    if layer == "bronze":
        checks["has_metadata"] = True
        checks["quality_score"] = 1.0 if row_count > 0 else 0.0

    elif layer == "silver":
        cols_df = query_duckdb(
            "SELECT AVG(_quality_score) AS avg_quality FROM data_table",
            {"data_table": parquet_path},
        )
        avg_q = float(cols_df["avg_quality"].iloc[0] or 0)
        checks["avg_quality_score"] = round(avg_q, 4)
        checks["quality_score"] = avg_q

    elif layer == "gold":
        null_df = query_duckdb(
            """
            SELECT
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_close,
                COUNT(*) AS total
            FROM data_table
            """,
            {"data_table": parquet_path},
        )
        total = int(null_df["total"].iloc[0])
        null_close = int(null_df["null_close"].iloc[0])
        completeness = 1.0 - (null_close / total if total else 0)
        checks["completeness"] = round(completeness, 4)
        checks["quality_score"] = completeness

    return checks


class LayerMonitor:
    """Mesure latence, cout (taille stockage) et qualite."""

    def __init__(self):
        self.metrics: list[dict] = []

    def measure_layer(
        self,
        layer: str,
        table: str,
        parquet_path: str,
        start_time: float,
    ) -> dict:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        table_dir = layer_table_dir(layer, table)
        size_bytes = dir_size(table_dir)
        quality = _quality_checks(layer, parquet_path)

        metric = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "table": table,
            "data_type": DATA_TYPES.get(table, "aggregated"),
            "latency_ms": latency_ms,
            "storage_bytes": size_bytes,
            "storage_mb": round(size_bytes / (1024 * 1024), 4),
            "estimated_cost_usd": round(size_bytes / (1024**3) * 0.023, 6),
            "quality": quality,
        }
        self.metrics.append(metric)
        return metric

    def save_report(self, batch_id: str) -> str:
        report = {
            "batch_id": batch_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "layers": self.metrics,
            "summary": self._summary(),
        }
        path = monitoring_report_path(batch_id)
        write_json(report, path)
        self._append_history(batch_id)
        return path

    def _append_history(self, batch_id: str) -> None:
        df = self.to_dataframe()
        if df.empty:
            return
        df["batch_id"] = batch_id
        df["recorded_at"] = datetime.now(timezone.utc).isoformat()
        history_path = MONITORING_HISTORY_PATH
        if exists(history_path):
            existing = read_parquet(history_path)
            df = pd.concat([existing, df], ignore_index=True)
        write_parquet(df, history_path)

    def _summary(self) -> dict:
        if not self.metrics:
            return {}
        by_layer: dict[str, list] = {}
        for m in self.metrics:
            by_layer.setdefault(m["layer"], []).append(m)

        summary = {}
        for layer, items in by_layer.items():
            summary[layer] = {
                "total_latency_ms": sum(i["latency_ms"] for i in items),
                "total_storage_mb": round(sum(i["storage_mb"] for i in items), 4),
                "total_cost_usd": round(sum(i["estimated_cost_usd"] for i in items), 6),
                "avg_quality_score": round(
                    sum(i["quality"].get("quality_score", 0) for i in items) / len(items), 4
                ),
                "tables": len(items),
            }
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for m in self.metrics:
            rows.append({
                "layer": m["layer"],
                "table": m["table"],
                "data_type": m.get("data_type", "aggregated"),
                "latency_ms": m["latency_ms"],
                "storage_mb": m["storage_mb"],
                "cost_usd": m["estimated_cost_usd"],
                "quality_score": m["quality"].get("quality_score", 0),
                "row_count": m["quality"].get("row_count", 0),
            })
        return pd.DataFrame(rows)
