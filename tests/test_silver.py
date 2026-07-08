"""Tests couche Silver."""

import pandas as pd

from src.bronze.ingest import run_bronze_ingestion
from src.silver.transform import run_silver_transform
from src.storage.io import read_parquet


def test_silver_transform(project_root):
    run_bronze_ingestion("test-batch")
    result = run_silver_transform("test-batch")

    assert result["tables"]["stock_prices"]["rows"] == 3
    assert result["tables"]["news_reddit"]["rows"] == 4
    assert result["tables"]["news_combined"]["rows"] == 3

    combined = read_parquet(result["tables"]["news_combined"]["path"])
    assert "_combined_finance_ratio" in combined.columns
    assert combined["Label"].isin([0, 1]).all()
    assert combined["_quality_score"].mean() == 1.0
