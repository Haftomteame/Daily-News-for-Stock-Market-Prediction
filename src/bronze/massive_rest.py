"""Client REST Massive — barres OHLCV journalieres."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

from src.config import MASSIVE_API_BASE, MASSIVE_TICKER

MASSIVE_API_KEY_ENV = "MASSIVE_API_KEY"


class MassiveRestError(Exception):
    """Erreur API REST Massive (credentials, plan, reseau)."""


def has_massive_api_key() -> bool:
    return bool(os.getenv(MASSIVE_API_KEY_ENV))


def _api_key() -> str:
    key = os.getenv(MASSIVE_API_KEY_ENV)
    if not key:
        raise MassiveRestError(
            f"{MASSIVE_API_KEY_ENV} requis (Dashboard Massive > Keys > Accessing the API)."
        )
    return key


def fetch_daily_bars(
    ticker: str | None = None,
    start: date | None = None,
    end: date | None = None,
    adjusted: bool = True,
) -> pd.DataFrame:
    """GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}"""
    from src.bronze.massive import infer_date_range_from_news

    ticker = (ticker or MASSIVE_TICKER).upper()
    if start is None or end is None:
        inferred_start, inferred_end = infer_date_range_from_news()
        start = start or inferred_start
        end = end or inferred_end

    url: str | None = (
        f"{MASSIVE_API_BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    )
    params: dict | None = {
        "apiKey": _api_key(),
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
    }

    rows: list[dict] = []
    while url:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 403:
            raise MassiveRestError(
                "Acces refuse (403). Verifiez votre plan ou la plage d'historique "
                "(Stocks Basic = 2 ans max)."
            )
        if response.status_code == 401:
            raise MassiveRestError("Cle API invalide (401).")
        response.raise_for_status()

        payload = response.json()
        if payload.get("status") not in {None, "OK", "DELAYED"}:
            raise MassiveRestError(f"API status: {payload.get('status')}")

        for bar in payload.get("results") or []:
            rows.append({
                "Date": pd.to_datetime(bar["t"], unit="ms"),
                "Open": float(bar["o"]),
                "High": float(bar["h"]),
                "Low": float(bar["l"]),
                "Close": float(bar["c"]),
                "Volume": int(bar["v"]),
                "Adj Close": float(bar["c"]),
            })

        url = payload.get("next_url")
        params = None

    if not rows:
        raise MassiveRestError(
            f"Aucune barre pour {ticker} entre {start} et {end}."
        )

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
