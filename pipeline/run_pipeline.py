#!/usr/bin/env python3
"""Orchestrateur ELT — Bronze → Silver → Gold → ML avec monitoring."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv

load_dotenv()

from src.bronze.ingest import (
    ingest_massive_day_aggs,
    ingest_news_combined,
    ingest_news_reddit,
    ingest_stock_prices,
)
from src.config import (
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    MONITORING_HISTORY_PATH,
    gold_data_path,
)
from src.storage.io import storage_label
from src.gold.aggregate import run_gold_aggregate
from src.ml.train import run_ml_training
from src.monitoring.metrics import LayerMonitor
from src.silver.transform import (
    transform_news_combined,
    transform_news_reddit,
    transform_stock_prices,
)


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
    args = parser.parse_args()

    from datetime import datetime

    batch_id = str(uuid.uuid4())
    monitor = LayerMonitor()
    stock_source = "massive" if args.massive else "auto"
    reddit_from = datetime.strptime(args.reddit_from, "%Y-%m-%d").date() if args.reddit_from else None
    reddit_to = datetime.strptime(args.reddit_to, "%Y-%m-%d").date() if args.reddit_to else None

    print("=" * 60)
    print("  DATA LAKEHOUSE - Daily News / Stock Market")
    print(f"  Batch ID: {batch_id}")
    print(f"  Stockage   : {storage_label()}")
    if args.refresh_bronze:
        print(f"  Refresh    : oui (stock={'Massive' if args.massive else 'auto'})")
    else:
        print("  Refresh    : non (bronze existant)")
    print("=" * 60)

    # -- BRONZE --
    print("\n[BRONZE] Ingestion ELT (lakehouse/bronze/)...")
    bronze_tables = [
        (
            "stock_prices",
            lambda bid: ingest_stock_prices(
                bid, refresh=args.refresh_bronze, source=stock_source
            ),
        ),
        (
            "news_reddit",
            lambda bid: ingest_news_reddit(
                bid,
                refresh=args.refresh_bronze,
                date_from=reddit_from,
                date_to=reddit_to,
            ),
        ),
        ("news_combined", ingest_news_combined),
    ]
    if args.massive_raw:
        bronze_tables.append(("massive_day_aggs", ingest_massive_day_aggs))
    for table, fn in bronze_tables:
        t0 = time.perf_counter()
        df, path = fn(batch_id)
        monitor.measure_layer("bronze", table, path, t0)
        print(f"  OK {table}: {len(df):,} lignes")

    # -- SILVER --
    print("\n[SILVER] Nettoyage + metadata...")
    silver_tables = [
        ("stock_prices", transform_stock_prices),
        ("news_reddit", transform_news_reddit),
        ("news_combined", transform_news_combined),
    ]
    for table, fn in silver_tables:
        t0 = time.perf_counter()
        df, path = fn(batch_id)
        monitor.measure_layer("silver", table, path, t0)
        print(f"  OK {table}: {len(df):,} lignes")

    # -- GOLD --
    print("\n[GOLD] Agregation KPIs (schema fixe)...")
    t0 = time.perf_counter()
    gold_result = run_gold_aggregate(batch_id)
    monitor.measure_layer(
        "gold",
        gold_result["table"],
        gold_result["path"],
        t0,
    )
    print(f"  OK {gold_result['table']}: {gold_result['rows']:,} lignes")

    # -- ML --
    print("\n[ML] Entrainement modele (Combined_News_DJIA + Gold)...")
    t0 = time.perf_counter()
    ml_result = run_ml_training(batch_id, prediction_year=args.predict_year)
    metrics = ml_result["metrics"]
    print(
        f"  OK annee prediction={ml_result['prediction_year']}  "
        f"train={metrics['samples_train']:,}  predict={metrics['samples_predict']:,}"
    )
    print(f"  OK accuracy={metrics['accuracy']:.1%}  f1={metrics['f1']:.1%}")
    print(f"  OK predictions: {ml_result['rows']:,} lignes")

    # -- MONITORING --
    report_path = monitor.save_report(batch_id)
    summary = monitor._summary()

    print("\n" + "=" * 60)
    print("  MONITORING - Resume par couche")
    print("=" * 60)
    for layer, stats in summary.items():
        print(f"\n  [{layer.upper()}]")
        print(f"    Latence totale : {stats['total_latency_ms']:.0f} ms")
        print(f"    Stockage       : {stats['total_storage_mb']:.2f} MB")
        print(f"    Cout estime    : ${stats['total_cost_usd']:.6f}")
        print(f"    Qualite moy.   : {stats['avg_quality_score']:.2%}")

    print(f"\n  Rapport JSON     : {report_path}")
    print(f"  Historique       : {MONITORING_HISTORY_PATH}")
    print(f"  Couche Gold      : {gold_data_path()}")
    print(f"  Modele ML        : {ML_MODEL_PATH}")
    print(f"  Dashboard        : python scripts/run_dashboard.py  (http://localhost:8502)")
    print("\nPipeline termine avec succes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
