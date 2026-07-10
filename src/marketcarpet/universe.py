"""Univers de titres pour Market Carpet (S&P 500, actions populaires)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "dashboard" / "data" / "sp500_universe.json"

_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def load_universe_catalog() -> dict:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Univers Market Carpet introuvable : {UNIVERSE_PATH}")
    return json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))


def list_universe_groups() -> list[tuple[str, str]]:
    catalog = load_universe_catalog()
    return [
        (key, meta.get("label", key))
        for key, meta in catalog.get("groups", {}).items()
    ]


def load_universe(group_key: str = "sp500") -> pd.DataFrame:
    """Retourne symbol, name, sector, market_cap_b pour un groupe."""
    catalog = load_universe_catalog()
    groups = catalog.get("groups", {})
    if group_key not in groups:
        raise KeyError(f"Groupe inconnu : {group_key}")

    rows = []
    for item in groups[group_key].get("symbols", []):
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol or not _SYMBOL_RE.match(symbol):
            continue
        rows.append({
            "symbol": symbol,
            "name": str(item.get("name", symbol)),
            "sector": str(item.get("sector", "Autre")),
            "market_cap_b": float(item.get("market_cap_b", 1.0) or 1.0),
        })

    if not rows:
        return pd.DataFrame(columns=["symbol", "name", "sector", "market_cap_b"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
    return df.sort_values(["sector", "symbol"]).reset_index(drop=True)
