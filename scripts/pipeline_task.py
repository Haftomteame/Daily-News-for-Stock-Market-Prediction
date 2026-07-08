#!/usr/bin/env python3
"""Point d'entree CLI pour une tache pipeline (utilise par Airflow et scripts)."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv

load_dotenv()

from src.pipeline.lakehouse_tasks import (  # noqa: E402
    PipelineContext,
    run_bronze,
    run_gold_python,
    run_gold_spark,
    run_ml,
    run_silver,
    save_monitoring,
    wait_hdfs_ready,
    write_postgres_warehouse,
)


def _parse_date(value: str | None) -> datetime.date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute une tache du pipeline lakehouse")
    parser.add_argument(
        "task",
        choices=[
            "wait_hdfs",
            "bronze",
            "silver",
            "gold_python",
            "gold_spark",
            "ml",
            "warehouse",
            "monitoring",
        ],
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--refresh-bronze", action="store_true")
    parser.add_argument("--massive", action="store_true")
    parser.add_argument("--massive-raw", action="store_true")
    parser.add_argument("--predict-year", type=int, default=None)
    parser.add_argument("--reddit-from", type=str, default=None)
    parser.add_argument("--reddit-to", type=str, default=None)
    args = parser.parse_args()

    ctx = PipelineContext(
        batch_id=args.batch_id or str(uuid.uuid4()),
        refresh_bronze=args.refresh_bronze,
        massive=args.massive,
        massive_raw=args.massive_raw,
        predict_year=args.predict_year,
        reddit_from=_parse_date(args.reddit_from),
        reddit_to=_parse_date(args.reddit_to),
    )

    handlers = {
        "wait_hdfs": wait_hdfs_ready,
        "bronze": lambda: run_bronze(ctx),
        "silver": lambda: run_silver(ctx),
        "gold_python": lambda: run_gold_python(ctx),
        "gold_spark": lambda: run_gold_spark(ctx),
        "ml": lambda: run_ml(ctx),
        "warehouse": lambda: write_postgres_warehouse(ctx),
        "monitoring": lambda: save_monitoring(ctx),
    }

    result = handlers[args.task]()
    print(json.dumps({"task": args.task, "batch_id": ctx.batch_id, "result": result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
