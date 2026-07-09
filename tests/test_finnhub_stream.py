"""Tests agregation OHLCV Finnhub."""

from src.bronze.finnhub_stream import CandleAgg, format_candle_date


def test_candle_agg_single_bucket():
    agg = CandleAgg(bucket_sec=60, bucket_mode="minute")
    base_ms = 1_720_512_000_000  # 2024-07-09 04:00:00 UTC

    assert agg.update_trade(100.0, 10.0, base_ms + 1_000) is None
    assert agg.update_trade(101.5, 5.0, base_ms + 30_000) is None

    finished = agg.update_trade(99.0, 2.0, base_ms + 61_000)
    assert finished is not None
    assert finished["Open"] == 100.0
    assert finished["High"] == 101.5
    assert finished["Low"] == 100.0
    assert finished["Close"] == 101.5
    assert finished["Volume"] == 15.0
    assert finished["Adj Close"] == finished["Close"]


def test_candle_agg_flush_partial():
    agg = CandleAgg(bucket_sec=60, bucket_mode="minute")
    base_ms = 1_720_512_000_000
    agg.update_trade(50.0, 1.0, base_ms)
    row = agg.flush_partial()
    assert row is not None
    assert row["Close"] == 50.0
    assert agg.flush_partial() is None


def test_format_candle_date_uses_new_york():
    # 2024-07-09 09:30:00 ET = ouverture reguliere
    ts_ms = 1_720_530_600_000
    assert format_candle_date(ts_ms).startswith("2024-07-09")


def test_candle_agg_daily_bucket():
    agg = CandleAgg(bucket_mode="day")
    # Deux trades le meme jour NY
    day_ms = 1_720_530_600_000  # 2024-07-09 matin ET
    assert agg.update_trade(100.0, 10.0, day_ms) is None
    assert agg.update_trade(102.0, 5.0, day_ms + 3_600_000) is None
    row = agg.current_row()
    assert row is not None
    assert row["Open"] == 100.0
    assert row["High"] == 102.0
    assert row["Close"] == 102.0
    assert row["Volume"] == 15.0
    assert row["Date"].startswith("2024-07-09")


def test_bronze_table_for_mode():
    from src.bronze.finnhub_stream import bronze_table_for_mode

    assert bronze_table_for_mode("day") == "stock_prices"
    assert bronze_table_for_mode("minute") == "stock_prices_1m"


def test_append_bronze_rows(project_root):
    from src.bronze.storage import append_bronze_rows, read_bronze_data

    rows = [{
        "Date": "2026-07-08 10:00:00",
        "Open": 530.0,
        "High": 531.0,
        "Low": 529.5,
        "Close": 530.5,
        "Volume": 1200.0,
        "Adj Close": 530.5,
    }]
    append_bronze_rows(rows, "stock_prices_1m", "batch-1", "finnhub_test")
    df = read_bronze_data("stock_prices_1m")
    assert len(df) == 1
    assert float(df.iloc[0]["Close"]) == 530.5

    append_bronze_rows([{
        "Date": "2026-07-08 10:00:00",
        "Open": 530.0,
        "High": 532.0,
        "Low": 529.0,
        "Close": 531.0,
        "Volume": 1500.0,
        "Adj Close": 531.0,
    }], "stock_prices_1m", "batch-2", "finnhub_test")
    df = read_bronze_data("stock_prices_1m")
    assert len(df) == 1
    assert float(df.iloc[0]["Close"]) == 531.0
