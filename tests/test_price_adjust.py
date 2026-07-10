"""Tests ajustement des prix pour fractionnements."""

from __future__ import annotations

import pandas as pd

from src.finance.price_adjust import (
    adjust_ohlcv_for_splits,
    cumulative_split_factors,
    dedupe_splits,
    split_price_ratio,
)


def test_split_price_ratio():
    assert split_price_ratio(20, 1) == 20.0
    assert split_price_ratio(2, 1) == 2.0
    assert split_price_ratio(1, 1) == 1.0


def test_dedupe_splits_keeps_latest_ex_date_per_ratio():
    splits = pd.DataFrame({
        "act_symbol": ["GOOGL", "GOOGL"],
        "ex_date": ["2022-06-30", "2022-07-18"],
        "to_factor": [20.0, 20.0],
        "for_factor": [1.0, 1.0],
    })
    out = dedupe_splits(splits)
    assert len(out) == 1
    assert out.iloc[0]["ex_date"] == pd.Timestamp("2022-07-18")


def test_adjust_googl_like_series():
    ohlcv = pd.DataFrame({
        "date": pd.to_datetime(["2022-07-15", "2022-07-18", "2014-04-02", "2014-04-03"]),
        "open": [2200.0, 108.0, 1130.0, 570.0],
        "high": [2250.0, 115.0, 1140.0, 575.0],
        "low": [2190.0, 105.0, 1120.0, 565.0],
        "close": [2235.55, 109.03, 1135.10, 571.50],
        "volume": [1_000_000, 20_000_000, 2_000_000, 4_000_000],
    })
    splits = pd.DataFrame({
        "act_symbol": ["GOOG", "GOOGL"],
        "ex_date": ["2014-04-02", "2022-07-18"],
        "to_factor": [2.0, 20.0],
        "for_factor": [1.0, 1.0],
    })

    adjusted = adjust_ohlcv_for_splits(ohlcv, splits)

    pre_2022 = adjusted.loc[adjusted["date"] == "2022-07-15", "close"].iloc[0]
    post_2022 = adjusted.loc[adjusted["date"] == "2022-07-18", "close"].iloc[0]
    assert abs(pre_2022 - post_2022) < 15

    pre_2014 = adjusted.loc[adjusted["date"] == "2014-04-02", "close"].iloc[0]
    post_2014 = adjusted.loc[adjusted["date"] == "2014-04-03", "close"].iloc[0]
    assert abs(pre_2014 - post_2014) < 10


def test_cumulative_split_factors_before_and_after_ex_date():
    ohlcv = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "close": [400.0, 100.0, 102.0],
    })
    splits = dedupe_splits(pd.DataFrame({
        "act_symbol": ["TEST"],
        "ex_date": ["2020-01-02"],
        "to_factor": [4.0],
        "for_factor": [1.0],
    }))
    factors = cumulative_split_factors(ohlcv["date"], splits, ohlcv=ohlcv)
    assert factors.tolist() == [4.0, 1.0, 1.0]
