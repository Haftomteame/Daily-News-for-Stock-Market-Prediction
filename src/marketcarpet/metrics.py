"""Indicateurs Market Carpet (performance, RSI, jours haussiers/baissiers)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_DAYS = {
    "1D": 1,
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

MEASUREMENT_LABELS = {
    "performance": "Performance (%)",
    "rsi": "RSI (14 jours)",
    "up_down_days": "Jours haussiers − baissiers",
}


def _rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.tail(period).mean()
    avg_loss = losses.tail(period).mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _performance_pct(closes: pd.Series, trading_days: int) -> float | None:
    if len(closes) < trading_days + 1:
        return None
    start = closes.iloc[-(trading_days + 1)]
    end = closes.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return float((end / start - 1) * 100)


def _up_minus_down(closes: pd.Series, trading_days: int) -> float | None:
    window = closes.tail(trading_days + 1)
    if len(window) < 2:
        return None
    diff = window.diff().dropna()
    return float((diff > 0).sum() - (diff < 0).sum())


def compute_symbol_metrics(
    panel: pd.DataFrame,
    *,
    measurement: str,
    period_key: str,
) -> pd.DataFrame:
    """Calcule la métrique couleur par symbole."""
    trading_days = PERIOD_DAYS.get(period_key, 1)
    rows: list[dict] = []

    for symbol, grp in panel.groupby("symbol"):
        closes = grp.sort_values("date")["close"].astype(float)
        if closes.empty:
            continue

        last_close = float(closes.iloc[-1])
        last_date = grp["date"].max()

        if measurement == "performance":
            value = _performance_pct(closes, trading_days)
        elif measurement == "rsi":
            if period_key == "1D":
                value = _rsi(closes, 14)
            else:
                # Variation RSI sur la période (mode Period StockCharts)
                if len(closes) < trading_days + 15:
                    value = None
                else:
                    rsi_now = _rsi(closes, 14)
                    past = closes.iloc[: -(trading_days or 1)]
                    rsi_then = _rsi(past, 14)
                    value = (
                        None
                        if rsi_now is None or rsi_then is None
                        else rsi_now - rsi_then
                    )
        elif measurement == "up_down_days":
            value = _up_minus_down(closes, trading_days)
        else:
            value = _performance_pct(closes, trading_days)

        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
        change_abs = (
            last_close - prev_close
            if prev_close is not None and measurement == "performance"
            else None
        )

        rows.append({
            "symbol": symbol,
            "metric_value": value,
            "last_close": last_close,
            "last_date": last_date,
            "prev_close": prev_close,
            "change_abs": change_abs,
        })

    return pd.DataFrame(rows)


def color_range_for_measurement(measurement: str, period_key: str) -> tuple[float, float]:
    """Bornes symétriques pour l'échelle de couleur."""
    if measurement == "rsi" and period_key == "1D":
        return 30.0, 70.0
    if measurement == "up_down_days":
        days = PERIOD_DAYS.get(period_key, 5)
        return -days, days
    if period_key == "1D":
        return -3.0, 3.0
    if period_key == "5D":
        return -5.0, 5.0
    if period_key == "1M":
        return -8.0, 8.0
    if period_key in {"3M", "6M"}:
        return -15.0, 15.0
    return -25.0, 25.0
