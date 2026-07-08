"""Tests couche Gold."""

import pandas as pd

from src.bronze.ingest import run_bronze_ingestion
from src.config import GOLD_SCHEMA
from src.gold.aggregate import run_gold_aggregate
from src.silver.transform import run_silver_transform
from src.storage.io import read_parquet


def test_gold_aggregate(project_root):
    run_bronze_ingestion("test-batch")
    run_silver_transform("test-batch")
    result = run_gold_aggregate("test-batch")

    assert result["rows"] == 3
    df = read_parquet(result["path"])
    expected_cols = [col for col, _ in GOLD_SCHEMA] + ["_batch_id", "_layer"]
    assert list(df.columns) == expected_cols
    assert df["_layer"].iloc[0] == "gold"
