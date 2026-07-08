"""
DAG Airflow — Pipeline lakehouse Daily News / Stock Market Prediction.

Flux :
  init_batch → wait_hdfs → bronze → silver → gold (Spark ou Python)
  → ml_train → postgres_warehouse → monitoring_report

Variables d'environnement (docker-compose, profil airflow) :
  GOLD_ENGINE=spark|python   (defaut: spark)
  PIPELINE_MASSIVE=true      (defaut: true)
  PIPELINE_REFRESH_BRONZE=false
  PIPELINE_PREDICT_YEAR=2026
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "datax",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

GOLD_ENGINE = os.getenv("GOLD_ENGINE", "spark").lower()
PIPELINE_MASSIVE = os.getenv("PIPELINE_MASSIVE", "true").lower() in {"1", "true", "yes"}
PIPELINE_REFRESH_BRONZE = os.getenv("PIPELINE_REFRESH_BRONZE", "false").lower() in {
    "1",
    "true",
    "yes",
}
PIPELINE_PREDICT_YEAR = int(os.getenv("PIPELINE_PREDICT_YEAR", "2026"))
PIPELINE_STATE_KEY = "pipeline_state"


def _setup_project_path() -> None:
    import sys
    from pathlib import Path

    project_root = Path("/opt/airflow/project")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _build_context(ti):
    """Reconstruit le contexte pipeline depuis XCom."""
    _setup_project_path()

    from src.env import load_dotenv
    from src.pipeline.lakehouse_tasks import PipelineContext, restore_monitor

    load_dotenv("/opt/airflow/project/.env")

    state = ti.xcom_pull(key=PIPELINE_STATE_KEY) or {}
    batch_id = state.get("batch_id") or ti.xcom_pull(task_ids="init_batch", key="batch_id")
    ctx = PipelineContext(
        batch_id=batch_id,
        refresh_bronze=PIPELINE_REFRESH_BRONZE,
        massive=PIPELINE_MASSIVE,
        predict_year=PIPELINE_PREDICT_YEAR,
        gold_engine=GOLD_ENGINE,
    )
    restore_monitor(ctx, state.get("monitor_metrics"))
    return ctx


def _push_context(ti, ctx) -> None:
    from src.pipeline.lakehouse_tasks import pipeline_state

    ti.xcom_push(key=PIPELINE_STATE_KEY, value=pipeline_state(ctx))


def task_init_batch(**context):
    batch_id = str(uuid.uuid4())
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    context["ti"].xcom_push(key=PIPELINE_STATE_KEY, value={"batch_id": batch_id, "monitor_metrics": []})
    print(f"Batch ID: {batch_id}")
    return batch_id


def task_wait_hdfs(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import wait_hdfs_ready

    wait_hdfs_ready()
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "status": "hdfs_ready"}


def task_bronze(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import run_bronze

    result = run_bronze(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "tables": list(result.keys())}


def task_silver(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import run_silver

    result = run_silver(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "tables": list(result.keys())}


def task_gold(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import run_gold_python, run_gold_spark

    if GOLD_ENGINE == "spark":
        result = run_gold_spark(ctx)
    else:
        result = run_gold_python(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "gold": result}


def task_ml(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import run_ml

    result = run_ml(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "metrics": result["metrics"]}


def task_warehouse(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import write_postgres_warehouse

    result = write_postgres_warehouse(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "warehouse": result}


def task_monitoring(**context):
    ti = context["ti"]
    ctx = _build_context(ti)
    from src.pipeline.lakehouse_tasks import save_monitoring

    result = save_monitoring(ctx)
    _push_context(ti, ctx)
    return {"batch_id": ctx.batch_id, "report": result["report_path"]}


with DAG(
    dag_id="stock_market_lakehouse",
    default_args=DEFAULT_ARGS,
    description="Pipeline Medallion Bronze → Silver → Gold → ML → PostgreSQL",
    schedule_interval=os.getenv("AIRFLOW_DAG_SCHEDULE", "@daily"),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lakehouse", "stock-market", "hdfs", "spark"],
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    init_batch = PythonOperator(task_id="init_batch", python_callable=task_init_batch)
    wait_hdfs = PythonOperator(task_id="wait_hdfs", python_callable=task_wait_hdfs)
    bronze = PythonOperator(task_id="bronze_ingest", python_callable=task_bronze)
    silver = PythonOperator(task_id="silver_transform", python_callable=task_silver)
    gold = PythonOperator(task_id=f"gold_{GOLD_ENGINE}", python_callable=task_gold)
    ml_train = PythonOperator(task_id="ml_train", python_callable=task_ml)
    postgres_warehouse = PythonOperator(
        task_id="postgres_warehouse",
        python_callable=task_warehouse,
    )
    monitoring_report = PythonOperator(
        task_id="monitoring_report",
        python_callable=task_monitoring,
    )

    start >> init_batch >> wait_hdfs >> bronze >> silver >> gold >> ml_train
    ml_train >> postgres_warehouse >> monitoring_report >> end
