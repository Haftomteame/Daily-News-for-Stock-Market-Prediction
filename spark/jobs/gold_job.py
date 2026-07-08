#!/usr/bin/env python3
"""Job Spark Gold : Silver (HDFS Parquet) -> Gold (HDFS Parquet).

Option A : Spark pour ETL (Silver/Gold), ML reste en scikit-learn.
"""

from __future__ import annotations

import os
import uuid

from pyspark.sql import SparkSession, functions as F, Window


def main() -> int:
    hdfs_base = os.getenv("HDFS_BASE_PATH", "/datax").rstrip("/")
    default_fs = os.getenv("HDFS_DEFAULT_FS", "hdfs://namenode:8020")

    silver_stock = f"{default_fs}{hdfs_base}/lakehouse/silver/stock_prices/data.parquet"
    silver_news = f"{default_fs}{hdfs_base}/lakehouse/silver/news_reddit/data.parquet"
    tmp_id = uuid.uuid4().hex
    gold_out_file_uri = f"{default_fs}{hdfs_base}/lakehouse/gold/daily_market_kpis/data.parquet"
    gold_tmp_dir_uri = f"{default_fs}{hdfs_base}/lakehouse/gold/daily_market_kpis/_tmp_{tmp_id}"

    # Paths "sans scheme" pour les operations Hadoop FS (base sur fs.defaultFS).
    gold_out_file_path = f"{hdfs_base}/lakehouse/gold/daily_market_kpis/data.parquet"
    gold_tmp_dir_path = f"{hdfs_base}/lakehouse/gold/daily_market_kpis/_tmp_{tmp_id}"

    spark = (
        SparkSession.builder.appName("datax-gold-job")
        .config("spark.hadoop.fs.defaultFS", default_fs)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    stock = spark.read.parquet(silver_stock).where(F.col("_quality_score") == 1)
    news = spark.read.parquet(silver_news).where(F.col("_quality_score") == 1)

    stock = (
        stock.withColumn("date", F.to_date("Date"))
        .select(
            "date",
            F.col("Open").cast("double").alias("open"),
            F.col("High").cast("double").alias("high"),
            F.col("Low").cast("double").alias("low"),
            F.col("Close").cast("double").alias("close"),
            F.col("Volume").cast("long").alias("volume"),
        )
        .dropna(subset=["date"])
        .dropDuplicates(["date"])
    )

    w = Window.orderBy("date")
    stock = stock.withColumn("prev_close", F.lag("close").over(w))
    stock = stock.withColumn(
        "daily_return_pct",
        F.when(F.col("prev_close").isNull(), F.lit(None)).otherwise(
            (F.col("close") - F.col("prev_close")) / F.col("prev_close") * F.lit(100.0)
        ),
    )
    stock = stock.withColumn(
        "volatility_5d",
        F.stddev("close").over(w.rowsBetween(-4, 0)),
    )

    news_agg = (
        news.withColumn("date", F.to_date("Date"))
        .groupBy("date")
        .agg(
            F.count(F.lit(1)).cast("int").alias("news_count"),
            F.avg(F.col("_headline_length").cast("double")).alias("avg_headline_length"),
            F.avg(F.when(F.col("_has_finance_keyword") == True, F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "finance_news_ratio"
            ),
        )
    )

    df = (
        stock.join(news_agg, on="date", how="left")
        .withColumn("news_count", F.coalesce(F.col("news_count"), F.lit(0)))
        .withColumn("avg_headline_length", F.round(F.coalesce(F.col("avg_headline_length"), F.lit(0.0)), 2))
        .withColumn("finance_news_ratio", F.round(F.coalesce(F.col("finance_news_ratio"), F.lit(0.0)), 4))
        .withColumn("daily_return_pct", F.round(F.col("daily_return_pct"), 4))
        .withColumn("volatility_5d", F.round(F.col("volatility_5d"), 4))
        .withColumn(
            "market_direction",
            F.when(F.col("daily_return_pct") > F.lit(0.1), F.lit("UP"))
            .when(F.col("daily_return_pct") < F.lit(-0.1), F.lit("DOWN"))
            .otherwise(F.lit("FLAT")),
        )
        .withColumn("computed_at", F.current_timestamp())
        .orderBy("date")
    )

    # Spark ecrit un repertoire; notre projet attend un fichier unique `data.parquet`.
    # On ecrit en temp (coalesce(1)) puis on renomme le seul part-*.parquet en `data.parquet`.
    df.coalesce(1).write.mode("overwrite").parquet(gold_tmp_dir_uri)

    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    hconf.set("fs.defaultFS", default_fs)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)

    tmp_path = jvm.org.apache.hadoop.fs.Path(gold_tmp_dir_path)
    out_path = jvm.org.apache.hadoop.fs.Path(gold_out_file_path)

    # Si une ancienne sortie existe en dossier (erreur historique), on supprime.
    if fs.exists(out_path) and fs.isDirectory(out_path):
        fs.delete(out_path, True)

    # Supprime l'ancien fichier s'il existe.
    if fs.exists(out_path) and fs.isFile(out_path):
        fs.delete(out_path, False)

    statuses = fs.listStatus(tmp_path)
    part_files = [
        st.getPath() for st in statuses
        if st.isFile() and st.getPath().getName().startswith("part-") and st.getPath().getName().endswith(".parquet")
    ]
    if len(part_files) != 1:
        raise RuntimeError(f"Attendu 1 part file, trouve {len(part_files)} dans {gold_tmp_dir_uri}")

    fs.mkdirs(out_path.getParent())
    fs.rename(part_files[0], out_path)
    fs.delete(tmp_path, True)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

