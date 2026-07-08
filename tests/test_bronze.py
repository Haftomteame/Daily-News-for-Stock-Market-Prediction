"""Tests couche Bronze."""

from src.bronze.ingest import run_bronze_ingestion
from src.storage.io import exists


def test_bronze_ingestion(project_root):
    result = run_bronze_ingestion("test-batch")
    assert result["batch_id"] == "test-batch"
    assert result["tables"]["stock_prices"]["rows"] == 3
    assert result["tables"]["news_reddit"]["rows"] == 4
    assert result["tables"]["news_combined"]["rows"] == 3

    for table in result["tables"].values():
        assert exists(table["path"])
