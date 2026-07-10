import os
from pathlib import Path

from src.env import PROJECT_ROOT, load_dotenv

load_dotenv()

# Dossier legacy (migration ponctuelle CSV -> lakehouse uniquement).
LEGACY_DATA_DIR = PROJECT_ROOT / "Data"

# Chemins logiques — resolus via src/storage (local ou HDFS selon STORAGE_BACKEND).
from src.storage.paths import (  # noqa: E402
    bronze_parquet,
    gold_parquet,
    massive_cache_dir,
    ml_metrics_path,
    ml_model_path,
    ml_predictions_path,
    monitoring_history_path,
    silver_parquet,
)

ML_PREDICTION_YEAR = 2026

TRAINING_SYMBOLS = [
    "DIA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "XOM",
]

MASSIVE_CACHE_DIR = massive_cache_dir()
MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_S3_BUCKET = "flatfiles"
MASSIVE_API_BASE = "https://api.massive.com"
MASSIVE_TICKER = "DIA"
FINNHUB_TICKER = os.getenv("FINNHUB_SYMBOL", MASSIVE_TICKER)
FINNHUB_BUCKET_SEC = int(os.getenv("FINNHUB_BUCKET_SEC", "60"))
FINNHUB_BUCKET_MODE = os.getenv("FINNHUB_BUCKET_MODE", "minute").lower()

DATA_TYPES = {
    "stock_prices": "structured",
    "news_reddit": "unstructured",
    "news_combined": "hybrid",
}

BRONZE_TABLES = {
    "stock_prices": "stock_prices",
    "stock_prices_1m": "stock_prices_1m",
    "news_reddit": "news_reddit",
    "news_combined": "news_combined",
    "massive_day_aggs": "massive_day_aggs",
}

SILVER_TABLES = {
    "stock_prices": "stock_prices",
    "news_reddit": "news_reddit",
    "news_combined": "news_combined",
}

GOLD_TABLES = {
    "daily_market_kpis": "daily_market_kpis",
}

ML_MODEL_PATH = ml_model_path()
ML_METRICS_PATH = ml_metrics_path()
ML_PREDICTIONS_PATH = ml_predictions_path()
MONITORING_HISTORY_PATH = monitoring_history_path()


def bronze_data_path(table: str) -> str:
    return bronze_parquet(table)


def silver_data_path(table: str) -> str:
    return silver_parquet(table)


def gold_data_path() -> str:
    return gold_parquet()


GOLD_SCHEMA = [
    ("date", "DATE"),
    ("open", "DOUBLE"),
    ("high", "DOUBLE"),
    ("low", "DOUBLE"),
    ("close", "DOUBLE"),
    ("volume", "BIGINT"),
    ("daily_return_pct", "DOUBLE"),
    ("volatility_5d", "DOUBLE"),
    ("news_count", "INTEGER"),
    ("avg_headline_length", "DOUBLE"),
    ("finance_news_ratio", "DOUBLE"),
    ("market_direction", "VARCHAR"),
    ("computed_at", "TIMESTAMP"),
]
