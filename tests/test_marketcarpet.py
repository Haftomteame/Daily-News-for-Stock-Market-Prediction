"""Tests Market Carpet."""

from __future__ import annotations

import pandas as pd

from src.marketcarpet.metrics import _performance_pct, _rsi, compute_symbol_metrics
from src.marketcarpet.universe import load_universe


def test_load_universe_sp500():
    df = load_universe("sp500")
    assert not df.empty
    assert "sector" in df.columns
    assert df["symbol"].is_unique


def test_performance_pct():
    closes = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert abs(_performance_pct(closes, 1) - (103 / 102 - 1) * 100) < 0.01
    assert abs(_performance_pct(closes, 3) - (103 / 100 - 1) * 100) < 0.01


def test_compute_symbol_metrics():
    panel = pd.DataFrame({
        "symbol": ["AAA"] * 30 + ["BBB"] * 30,
        "date": pd.date_range("2024-01-01", periods=30).tolist() * 2,
        "close": list(range(100, 130)) + list(range(200, 170, -1)),
        "volume": [1_000_000] * 60,
    })
    out = compute_symbol_metrics(panel, measurement="performance", period_key="5D")
    assert len(out) == 2
    assert out["metric_value"].notna().all()
