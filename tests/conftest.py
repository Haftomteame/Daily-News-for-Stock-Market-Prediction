"""Fixtures pytest — mini dataset isole (stockage local temporaire)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path))

    import importlib
    import src.config as config
    import src.storage.io as storage_io
    import src.storage.paths as storage_paths

    importlib.reload(storage_io)
    importlib.reload(storage_paths)
    importlib.reload(config)

    stock_df = pd.DataFrame({
        "Date": ["2016-07-01", "2016-06-30", "2016-06-29"],
        "Open": [17924.0, 17712.0, 17456.0],
        "High": [18002.0, 17930.0, 17704.0],
        "Low": [17916.0, 17711.0, 17456.0],
        "Close": [17949.0, 17929.0, 17694.0],
        "Volume": [82160000, 133030000, 106380000],
        "Adj Close": [17949.0, 17929.0, 17694.0],
    })
    reddit_df = pd.DataFrame({
        "Date": ["2016-07-01", "2016-07-01", "2016-06-30", "2016-06-29"],
        "News": [
            "Stock market hits record high",
            "Fed raises interest rates",
            "Dow Jones falls on earnings",
            "Wall Street gains on jobs data",
        ],
    })
    combined_df = pd.DataFrame({
        "Date": ["2016-07-01", "2016-06-30", "2016-06-29"],
        "Label": [1, 0, 1],
        "Top1": [
            "b'Stock market rally continues'",
            "b'Market crash fears grow'",
            "b'Wall Street gains on jobs data'",
        ],
        "Top2": ["b'Investor sentiment strong'", "b'Recession warning'", "b'Nasdaq hits high'"],
    })

    from src.bronze.storage import write_bronze

    write_bronze(stock_df, "stock_prices", "fixture-batch", "fixture")
    write_bronze(reddit_df, "news_reddit", "fixture-batch", "fixture")
    write_bronze(combined_df, "news_combined", "fixture-batch", "fixture")

    import src.bronze.ingest as bronze_ingest
    import src.gold.aggregate as gold_aggregate
    import src.ml.train as ml_train
    import src.monitoring.metrics as monitoring_metrics
    import src.silver.transform as silver_transform

    ml_train.ML_MODEL_PATH = config.ML_MODEL_PATH
    ml_train.ML_METRICS_PATH = config.ML_METRICS_PATH
    ml_train.ML_PREDICTIONS_PATH = config.ML_PREDICTIONS_PATH
    monitoring_metrics.MONITORING_HISTORY_PATH = config.MONITORING_HISTORY_PATH

    return tmp_path
