#!/usr/bin/env python3
"""Orchestrateur ELT — Bronze → Silver → Gold → ML avec monitoring."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv

load_dotenv()

from src.config import (
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    MONITORING_HISTORY_PATH,
    gold_data_path,
)
from src.pipeline.lakehouse_tasks import PipelineContext, run_full_pipeline
from src.storage.io import storage_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline lakehouse Bronze → Gold → ML")
    parser.add_argument(
        "--massive",
        action="store_true",
        help="Rafraichir stock_prices via Massive (cache S3 > REST API) avec --refresh-bronze",
    )
    parser.add_argument(
        "--massive-raw",
        action="store_true",
        help="Ingerer aussi les day aggregates bruts (bronze/massive_day_aggs)",
    )
    parser.add_argument(
        "--refresh-bronze",
        action="store_true",
        help="Re-fetch APIs et re-ecrit lakehouse/bronze/ (sinon lit le bronze existant)",
    )
    parser.add_argument(
        "--reddit-from",
        type=str,
        default=None,
        help="Debut fetch Reddit (YYYY-MM-DD) avec --refresh-bronze",
    )
    parser.add_argument(
        "--reddit-to",
        type=str,
        default=None,
        help="Fin fetch Reddit (YYYY-MM-DD) avec --refresh-bronze",
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=None,
        help="Annee de prediction out-of-sample (defaut: 2016, pas de donnees 2017)",
    )
    parser.add_argument(
        "--gold-engine",
        choices=["spark", "python"],
        default="python",
        help="Moteur Gold : spark (HDFS+Spark) ou python (DuckDB, defaut)",
    )
    args = parser.parse_args()

    reddit_from = datetime.strptime(args.reddit_from, "%Y-%m-%d").date() if args.reddit_from else None
    reddit_to = datetime.strptime(args.reddit_to, "%Y-%m-%d").date() if args.reddit_to else None

    print("=" * 60)
    print("  DATA LAKEHOUSE - Daily News / Stock Market")
    print(f"  Stockage   : {storage_label()}")
    print(f"  Gold       : {args.gold_engine}")
    if args.refresh_bronze:
        print(f"  Refresh    : oui (stock={'Massive' if args.massive else 'auto'})")
    else:
        print("  Refresh    : non (bronze existant)")
    print("=" * 60)

    ctx = run_full_pipeline(
        refresh_bronze=args.refresh_bronze,
        massive=args.massive,
        massive_raw=args.massive_raw,
        predict_year=args.predict_year,
        reddit_from=reddit_from,
        reddit_to=reddit_to,
        gold_engine=args.gold_engine,
    )

    summary = ctx.monitor._summary()
    print("\n" + "=" * 60)
    print("  MONITORING - Resume par couche")
    print("=" * 60)
    for layer, stats in summary.items():
        print(f"\n  [{layer.upper()}]")
        print(f"    Latence totale : {stats['total_latency_ms']:.0f} ms")
        print(f"    Stockage       : {stats['total_storage_mb']:.2f} MB")
        print(f"    Cout estime    : ${stats['total_cost_usd']:.6f}")
        print(f"    Qualite moy.   : {stats['avg_quality_score']:.2%}")

    print(f"\n  Batch ID         : {ctx.batch_id}")
    print(f"  Couche Gold      : {gold_data_path()}")
    print(f"  Modele ML        : {ML_MODEL_PATH}")
    print(f"  Historique       : {MONITORING_HISTORY_PATH}")
    print(f"  Dashboard        : python scripts/run_dashboard.py  (http://localhost:8502)")
    print("\nPipeline termine avec succes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
