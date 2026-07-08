"""Taches modulaires du pipeline lakehouse (reutilisables par CLI et Airflow)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.bronze.ingest import (
    ingest_massive_day_aggs,
    ingest_news_combined,
    ingest_news_reddit,
    ingest_stock_prices,
)
from src.config import gold_data_path
from src.gold.aggregate import run_gold_aggregate
from src.ml.train import run_ml_training
from src.monitoring.metrics import LayerMonitor
from src.silver.transform import (
    transform_news_combined,
    transform_news_reddit,
    transform_stock_prices,
)
from src.storage.io import read_parquet


@dataclass
class PipelineContext:
    """Contexte partage entre les taches du pipeline."""

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    refresh_bronze: bool = False
    massive: bool = False
    massive_raw: bool = False
    predict_year: int | None = None
    reddit_from: date | None = None
    reddit_to: date | None = None
    gold_engine: str = "spark"  # spark | python
    monitor: LayerMonitor = field(default_factory=LayerMonitor)


def _storage_backend() -> str:
    import os

    return os.getenv("STORAGE_BACKEND", "local").lower()


def wait_hdfs_ready() -> None:
    """Attend que le NameNode HDFS soit actif (no-op si stockage local)."""
    if _storage_backend() != "hdfs":
        return

    import os
    import time

    import requests

    host = os.getenv("HDFS_NAMENODE", "localhost")
    port = os.getenv("HDFS_WEB_PORT", "9870")
    url = f"http://{host}:{port}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus"
    timeout = int(os.getenv("HDFS_WAIT_SECONDS", "180"))

    for elapsed in range(timeout):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and "active" in response.text.lower():
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError(f"HDFS NameNode non disponible ({host}:{port})")


def run_bronze(ctx: PipelineContext) -> dict[str, Any]:
    """Ingestion Bronze (HDFS Parquet)."""
    stock_source = "massive" if ctx.massive else "auto"
    results: dict[str, Any] = {}

    tables = [
        (
            "stock_prices",
            lambda bid: ingest_stock_prices(bid, refresh=ctx.refresh_bronze, source=stock_source),
        ),
        (
            "news_reddit",
            lambda bid: ingest_news_reddit(
                bid,
                refresh=ctx.refresh_bronze,
                date_from=ctx.reddit_from,
                date_to=ctx.reddit_to,
            ),
        ),
        ("news_combined", ingest_news_combined),
    ]
    if ctx.massive_raw:
        tables.append(("massive_day_aggs", ingest_massive_day_aggs))

    for table, fn in tables:
        t0 = time.perf_counter()
        df, path = fn(ctx.batch_id)
        ctx.monitor.measure_layer("bronze", table, path, t0)
        results[table] = {"rows": len(df), "path": path}

    return results


def run_silver(ctx: PipelineContext) -> dict[str, Any]:
    """Transformation Silver."""
    results: dict[str, Any] = {}
    tables = [
        ("stock_prices", transform_stock_prices),
        ("news_reddit", transform_news_reddit),
        ("news_combined", transform_news_combined),
    ]
    for table, fn in tables:
        t0 = time.perf_counter()
        df, path = fn(ctx.batch_id)
        ctx.monitor.measure_layer("silver", table, path, t0)
        results[table] = {"rows": len(df), "path": path}
    return results


def run_gold_python(ctx: PipelineContext) -> dict[str, Any]:
    """Agregation Gold via DuckDB/Python."""
    t0 = time.perf_counter()
    result = run_gold_aggregate(ctx.batch_id)
    ctx.monitor.measure_layer("gold", result["table"], result["path"], t0)
    return {"table": result["table"], "rows": result["rows"], "path": result["path"]}


def run_gold_spark(ctx: PipelineContext) -> dict[str, Any]:
    """Agregation Gold via job Spark (container spark-gold ou fallback Python)."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    compose_file = project_root / "docker-compose.yml"
    docker_bin = shutil.which("docker")

    if docker_bin and compose_file.exists():
        cmd = [
            docker_bin,
            "compose",
            "-f",
            str(compose_file),
            "--profile",
            "hdfs",
            "--profile",
            "spark",
            "run",
            "--rm",
            "spark-gold",
        ]
        subprocess.run(cmd, check=True, cwd=project_root)
        t0 = time.perf_counter() - 0.001
        ctx.monitor.measure_layer("gold", "daily_market_kpis", gold_data_path(), t0)
        return {"engine": "spark", "path": gold_data_path()}

    if os.getenv("AIRFLOW_HOME"):
        raise RuntimeError(
            "Spark Gold requiert le socket Docker monte dans Airflow "
            "(/var/run/docker.sock) et les profils hdfs+spark actifs."
        )

    print("WARN : Docker indisponible — fallback Gold Python (DuckDB).")
    return run_gold_python(ctx)


def run_ml(ctx: PipelineContext) -> dict[str, Any]:
    """Entrainement ML + predictions."""
    result = run_ml_training(ctx.batch_id, prediction_year=ctx.predict_year)
    return {
        "prediction_year": result["prediction_year"],
        "metrics": result["metrics"],
        "rows": result["rows"],
    }


def write_postgres_warehouse(ctx: PipelineContext) -> dict[str, str]:
    """Charge Silver/Gold dans PostgreSQL (mode replace)."""
    from src.config import silver_data_path
    from src.db.postgres import pg_enabled, test_connection, write_replace

    if not pg_enabled():
        return {"status": "skipped", "reason": "PGHOST/PGPASSWORD non configures"}

    test_connection()
    write_replace("silver_stock_prices", read_parquet(silver_data_path("stock_prices")))
    write_replace("silver_news_reddit", read_parquet(silver_data_path("news_reddit")))
    write_replace("silver_news_combined", read_parquet(silver_data_path("news_combined")))
    write_replace("gold_daily_market_kpis", read_parquet(gold_data_path()))
    return {"status": "ok", "database": "wherehouse"}


def save_monitoring(ctx: PipelineContext) -> dict[str, Any]:
    """Sauvegarde le rapport de monitoring."""
    report_path = ctx.monitor.save_report(ctx.batch_id)
    summary = ctx.monitor._summary()
    return {"report_path": str(report_path), "summary": summary}


def pipeline_state(ctx: PipelineContext) -> dict[str, Any]:
    """Etat serialisable du pipeline (pour XCom Airflow)."""
    return {
        "batch_id": ctx.batch_id,
        "monitor_metrics": ctx.monitor.metrics,
    }


def restore_monitor(ctx: PipelineContext, metrics: list[dict] | None) -> None:
    """Restaure les metriques accumulees depuis une tache precedente."""
    if metrics:
        ctx.monitor.metrics = metrics


def run_full_pipeline(
    *,
    refresh_bronze: bool = False,
    massive: bool = False,
    massive_raw: bool = False,
    predict_year: int | None = None,
    reddit_from: date | None = None,
    reddit_to: date | None = None,
    gold_engine: str = "spark",
    batch_id: str | None = None,
) -> PipelineContext:
    """Execute le pipeline complet (meme logique que run_pipeline.py)."""
    ctx = PipelineContext(
        batch_id=batch_id or str(uuid.uuid4()),
        refresh_bronze=refresh_bronze,
        massive=massive,
        massive_raw=massive_raw,
        predict_year=predict_year,
        reddit_from=reddit_from,
        reddit_to=reddit_to,
        gold_engine=gold_engine,
    )

    if _storage_backend() == "hdfs":
        wait_hdfs_ready()
    run_bronze(ctx)
    run_silver(ctx)

    if ctx.gold_engine == "spark":
        run_gold_spark(ctx)
    else:
        run_gold_python(ctx)

    run_ml(ctx)
    write_postgres_warehouse(ctx)
    save_monitoring(ctx)
    return ctx
