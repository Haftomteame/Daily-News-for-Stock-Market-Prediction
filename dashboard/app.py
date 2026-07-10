#!/usr/bin/env python3
"""Dashboard — Prévisions bourse & actualités (interface grand public)."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FINNHUB_TICKER,
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    ML_PREDICTIONS_PATH,
    bronze_data_path,
    gold_data_path,
    silver_data_path,
)
from src.db.postgres import (  # noqa: E402
    DOLT_SCHEMA_DESCRIPTIONS,
    DOLT_SCHEMA_LABELS,
    dolt_table_label,
    pg_enabled,
    read_sql,
    symbol_row_count as pg_symbol_row_count,
    warehouse_table_stats,
)
from dashboard.market_carpet import render_market_carpet  # noqa: E402
from src.storage.io import (  # noqa: E402
    exists,
    glob_paths,
    monitoring_path,
    query_duckdb,
    read_json,
    read_joblib,
    read_parquet,
)
from src.env import load_dotenv  # noqa: E402
from src.storage.paths import (  # noqa: E402
    ml_metrics_path as _ml_metrics_path_for,
    ml_model_path as _ml_model_path_for,
    ml_predictions_path as _ml_predictions_path_for,
)

load_dotenv()

st.set_page_config(
    page_title="Prévisions Bourse & Actualités",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem; max-width: 1280px;
        background: linear-gradient(180deg, #f0f4ff 0%, #fafbff 12rem, #ffffff 24rem);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    /* Texte clair sur fond sombre (titres, labels, légendes) */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #e0e7ff !important;
    }
    /* Champs blancs — texte sombre lisible */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #1e1b4b !important;
        -webkit-text-fill-color: #1e1b4b !important;
        border-color: #c7d2fe !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] svg {
        color: #1e1b4b !important;
        fill: #1e1b4b !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
        color: #fff !important; border-radius: 10px; font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(255,255,255,.22); border-color: rgba(255,255,255,.35);
    }
    .hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 50%, #6366f1 100%);
        border-radius: 20px; padding: 2rem 2.25rem; margin-bottom: 1.5rem;
        color: #fff; box-shadow: 0 20px 50px rgba(67,56,202,.25);
    }
    .hero h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 .5rem; color: #fff; }
    .hero p  { color: rgba(255,255,255,.85); font-size: .95rem; margin: 0; line-height: 1.55; }
    .hero-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }
    .hero-pill {
        background: rgba(255,255,255,.15); backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
        padding: .35rem .85rem; font-size: .78rem; font-weight: 600;
    }
    .card {
        background: #fff; border: 1px solid #e8eaf6; border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px rgba(30,27,75,.06);
        height: 100%; transition: transform .2s, box-shadow .2s;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(67,56,202,.12);
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: .75rem;
    }
    .badge {
        font-size: .7rem; font-weight: 700; padding: .25rem .7rem;
        border-radius: 999px; text-transform: uppercase; letter-spacing: .04em;
    }
    .badge-volume  { background: #dbeafe; color: #1d4ed8; }
    .badge-social  { background: #ffedd5; color: #c2410c; }
    .badge-temps   { background: #ede9fe; color: #6d28d9; }
    .badge-ml      { background: #dcfce7; color: #15803d; }
    .badge-important { background: #dcfce7; color: #15803d; }
    .badge-insight { background: #ede9fe; color: #6d28d9; }
    .metric-value { font-size: 2.1rem; font-weight: 800; color: #1e1b4b; line-height: 1.1; }
    .metric-label { font-size: .92rem; color: #4b5563; margin-top: .2rem; font-weight: 500; }
    .metric-sub   { font-size: .78rem; color: #9ca3af; margin-top: .35rem; }
    .card-desc    { font-size: .85rem; color: #6b7280; margin-bottom: 1rem; line-height: 1.5; }
    .flow-node {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
        padding: .9rem 1rem; text-align: center; width: 100%;
        transition: border-color .2s, background .2s;
    }
    .flow-node.done   { border-color: #22c55e; background: linear-gradient(180deg,#f0fdf4,#fff); }
    .flow-node.wait   { border-color: #e5e7eb; background: #fafafa; }
    .flow-node .icon  { font-size: 1.6rem; line-height: 1; }
    .flow-node .name  { font-weight: 700; font-size: .88rem; color: #1f2937; margin-top: .35rem; }
    .flow-node .sub   { font-size: .72rem; color: #9ca3af; margin-top: .15rem; }
    .flow-arrow-h { color: #a5b4fc; font-size: 1.4rem; text-align: center; padding-top: 1.6rem; }
    .flow-arrow-v { color: #a5b4fc; font-size: 1.2rem; text-align: center; margin: .35rem 0; }
    .flow-ingest {
        background: linear-gradient(135deg,#eff6ff,#eef2ff);
        border: 1px solid #a5b4fc; border-radius: 14px;
        padding: 1rem 1.25rem; text-align: center; margin: 0 auto; max-width: 280px;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600; font-size: .92rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #4338ca !important; border-bottom-color: #4338ca !important;
    }
    .year-picker-caption {
        font-size: .82rem; color: #6b7280; margin: .35rem 0 .5rem;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

POSITIVE_KW = {
    "bull", "rally", "surge", "gain", "rise", "soar", "record", "growth",
    "jump", "climb", "boost", "high", "profit", "beat",
}
NEGATIVE_KW = {
    "crash", "fall", "drop", "bear", "decline", "loss", "plunge", "recession",
    "down", "slump", "tumble", "fear", "sell-off", "selloff", "warning",
}

FEATURE_LABELS = {
    "num__daily_return_pct": "Variation du marché (veille)",
    "num__volatility_5d": "Instabilité récente",
    "num__news_count": "Nombre d'actualités",
    "num__finance_news_ratio": "Actualités liées à la finance",
    "text__": "Ton des discussions Reddit",
}

DEFAULT_SYMBOL = FINNHUB_TICKER.upper()
POPULAR_SYMBOLS = [
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

SYMBOL_SCOPED_TABLES: dict[str, list[str]] = {
    "stocks": ["ohlcv", "dividend", "split"],
    "options": ["option_chain"],
    "earnings": [
        "earnings_calendar",
        "eps_estimate",
        "eps_history",
        "balance_sheet_assets",
        "balance_sheet_equity",
        "balance_sheet_liabilities",
        "cash_flow_statement",
        "income_statement",
        "rank_score",
        "sales_estimate",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_altair(chart: alt.Chart) -> None:
    """Affiche un graphique Altair (Vega-Lite v5) dans Streamlit."""
    st.altair_chart(chart, width="stretch", theme=None)


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _fmt_compact(value: int | float) -> str:
    n = float(value)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".0 M", " M")
    if n >= 1_000:
        return f"{n / 1_000:.0f} k"
    return f"{int(n):,}"


def _read_json_local_or_storage(path: str, local_rel: str) -> dict | None:
    if exists(path):
        try:
            return read_json(path)
        except Exception:
            pass
    local = PROJECT_ROOT / local_rel
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _row_count(report: dict | None, table: str) -> int:
    if not report:
        return 0
    for layer in report.get("layers", []):
        if layer.get("table") == table:
            return int(layer.get("quality", {}).get("row_count", 0))
    return 0


def _layer_ok(report: dict | None, layer_name: str) -> bool:
    if not report:
        return False
    items = [l for l in report.get("layers", []) if l.get("layer") == layer_name]
    if not items:
        return exists(gold_data_path()) if layer_name == "gold" else False
    return all(l.get("quality", {}).get("quality_score", 0) >= 0.8 for l in items)


def _classify_headline(text: str) -> str:
    lower = str(text).lower()
    if any(k in lower for k in POSITIVE_KW):
        return "Positif"
    if any(k in lower for k in NEGATIVE_KW):
        return "Négatif"
    return "Neutre"


def _safe_symbol(symbol: str | None) -> str:
    raw = (symbol or DEFAULT_SYMBOL).upper().strip()
    if re.fullmatch(r"[A-Z0-9.]{1,12}", raw):
        return raw
    return DEFAULT_SYMBOL


def _active_symbol(selected: str | None) -> str:
    return _safe_symbol(selected)


def _lakehouse_matches_symbol(symbol: str) -> bool:
    return symbol.upper() == DEFAULT_SYMBOL


def _symbol_row_count(schema: str, table: str, symbol: str) -> int:
    if not pg_enabled():
        return 0
    try:
        return pg_symbol_row_count(schema, table, symbol)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_all_monitoring_reports() -> list[dict]:
    reports: list[dict] = []
    seen: set[str] = set()
    try:
        for path in glob_paths(monitoring_path("report_*.json")):
            try:
                data = read_json(path)
                bid = str(data.get("batch_id", ""))
                if bid and bid not in seen:
                    reports.append(data)
                    seen.add(bid)
            except Exception:
                continue
    except Exception:
        pass
    local_dir = PROJECT_ROOT / "monitoring"
    if local_dir.exists():
        for path in local_dir.glob("report_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                bid = str(data.get("batch_id", ""))
                if bid and bid not in seen:
                    reports.append(data)
                    seen.add(bid)
            except Exception:
                continue
    reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return reports


@st.cache_data(ttl=60)
def load_gold() -> pd.DataFrame:
    path = gold_data_path()
    if not exists(path):
        return pd.DataFrame()
    try:
        df = read_parquet(path)
        if "date" in df.columns:
            df["date"] = _parse_date(df["date"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_ml_metrics(symbol: str | None = None) -> dict | None:
    symbol = _active_symbol(symbol)
    candidates = [
        _ml_metrics_path_for(symbol),
        f"lakehouse/ml/{symbol}/metrics.json",
    ]
    if symbol == DEFAULT_SYMBOL:
        candidates.append(ML_METRICS_PATH)
        candidates.append("lakehouse/ml/metrics.json")
    for path in candidates:
        local = f"lakehouse/ml/{symbol}/metrics.json" if symbol in path else "lakehouse/ml/metrics.json"
        data = _read_json_local_or_storage(path, local)
        if data:
            return data
    return None


@st.cache_data(ttl=60)
def load_ml_predictions(symbol: str | None = None) -> pd.DataFrame:
    symbol = _active_symbol(symbol)
    paths = [_ml_predictions_path_for(symbol)]
    if symbol == DEFAULT_SYMBOL:
        paths.append(ML_PREDICTIONS_PATH)
    for path in paths:
        if not exists(path):
            continue
        try:
            df = read_parquet(path)
            if "date" in df.columns:
                df["date"] = _parse_date(df["date"])
            return df.sort_values("date")
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sentiment_distribution(symbol: str | None = None) -> pd.DataFrame:
    path = silver_data_path("news_reddit")
    if not exists(path):
        path = bronze_data_path("news_reddit")
    if not exists(path):
        return pd.DataFrame()

    symbol = _active_symbol(symbol) if symbol else None
    symbol_filter = ""
    if symbol and not _lakehouse_matches_symbol(symbol):
        symbol_filter = f" AND UPPER(News) LIKE '%{symbol}%'"

    try:
        df = query_duckdb(
            f"""
            SELECT News FROM data_table
            WHERE News IS NOT NULL AND TRIM(CAST(News AS VARCHAR)) != ''
            {symbol_filter}
            LIMIT 50000
            """,
            {"data_table": path},
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    counts = df["News"].apply(_classify_headline).value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    order = ["Positif", "Neutre", "Négatif"]
    counts["sentiment"] = pd.Categorical(
        counts["sentiment"].astype(str), categories=order, ordered=True
    )
    return counts.sort_values("sentiment").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_feature_importance(symbol: str | None = None) -> pd.DataFrame:
    symbol = _active_symbol(symbol)
    paths = [_ml_model_path_for(symbol)]
    if symbol == DEFAULT_SYMBOL:
        paths.append(ML_MODEL_PATH)
    model_path = next((p for p in paths if exists(p)), None)
    if not model_path:
        return pd.DataFrame()
    try:
        model = read_joblib(model_path)
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
        names = prep.get_feature_names_out()
        coefs = abs(clf.coef_[0])
        rows = []
        for name, coef in zip(names, coefs, strict=False):
            label = name
            for prefix, friendly in FEATURE_LABELS.items():
                if name.startswith(prefix):
                    label = friendly if prefix != "text__" else f"Reddit: {name[6:][:18]}"
                    break
            rows.append({"feature": label, "importance": float(coef)})
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)

        # Regrouper les termes TF-IDF sous "Reddit_sentiment"
        numeric = df[~df["feature"].str.startswith("Reddit:")]
        text_total = df[df["feature"].str.startswith("Reddit:")]["importance"].sum()
        grouped = [
            {"feature": "Discussions sur Reddit", "importance": text_total},
            {"feature": "Actualités financières", "importance": _pick(numeric, "Actualités liées à la finance")},
            {"feature": "Volume d'actualités", "importance": _pick(numeric, "Nombre d'actualités")},
            {"feature": "Instabilité du marché", "importance": _pick(numeric, "Instabilité récente")},
            {"feature": "Variation de la veille", "importance": _pick(numeric, "Variation du marché (veille)")},
        ]
        out = pd.DataFrame(grouped)
        out = out[out["importance"] > 0].sort_values("importance", ascending=False)
        if out.empty:
            return df.head(5)
        max_val = out["importance"].max()
        out["pct"] = (out["importance"] / max_val).round(2)
        return out.head(5)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_symbol_sentiment_series(
    symbol: str,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Série journalière d'ambiance Reddit (gold ou titres filtrés par symbole)."""
    symbol = _active_symbol(symbol)
    if _lakehouse_matches_symbol(symbol):
        gold = load_gold()
        if gold.empty:
            return pd.DataFrame()
        df = gold[["date", "finance_news_ratio"]].copy()
        df = df.rename(columns={"finance_news_ratio": "sentiment"})
    else:
        path = silver_data_path("news_reddit")
        if not exists(path):
            path = bronze_data_path("news_reddit")
        if not exists(path):
            return pd.DataFrame()
        try:
            df = query_duckdb(
                f"""
                SELECT
                    CAST("Date" AS DATE) AS date,
                    COUNT(*) AS mention_count
                FROM data_table
                WHERE News IS NOT NULL
                  AND TRIM(CAST(News AS VARCHAR)) != ''
                  AND UPPER(News) LIKE '%{symbol}%'
                GROUP BY 1
                ORDER BY 1
                """,
                {"data_table": path},
            )
            if df.empty:
                return pd.DataFrame()
            df["date"] = _parse_date(df["date"])
            max_count = df["mention_count"].max() or 1
            df["sentiment"] = df["mention_count"] / max_count
        except Exception:
            return pd.DataFrame()

    if date_range:
        start, end = date_range
        df = df[
            (df["date"] >= pd.Timestamp(start))
            & (df["date"] <= pd.Timestamp(end))
        ]
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=600)
def load_symbol_name(symbol: str) -> str:
    symbol = _active_symbol(symbol)
    if not pg_enabled():
        return symbol
    try:
        df = read_sql(
            "SELECT security_name FROM stocks.symbol WHERE act_symbol = :symbol LIMIT 1",
            params={"symbol": symbol},
        )
        if not df.empty and pd.notna(df.iloc[0]["security_name"]):
            return str(df.iloc[0]["security_name"])
    except Exception:
        pass
    return symbol


@st.cache_data(ttl=300)
def load_symbol_dividends(symbol: str, limit: int = 6) -> pd.DataFrame:
    if not pg_enabled() or not symbol:
        return pd.DataFrame()
    try:
        return read_sql(
            """
            SELECT ex_date, amount
            FROM stocks.dividend
            WHERE act_symbol = :symbol
            ORDER BY ex_date DESC
            LIMIT :limit
            """,
            params={"symbol": symbol.upper(), "limit": limit},
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_stock_ohlcv(symbol: str, date_range: tuple | None = None) -> pd.DataFrame:
    """Cours journaliers d'un symbole depuis PostgreSQL (stocks.ohlcv)."""
    if not pg_enabled() or not symbol:
        return pd.DataFrame()
    symbol = symbol.upper()
    sql = """
        SELECT date, open, high, low, close, volume
        FROM stocks.ohlcv
        WHERE act_symbol = :symbol
    """
    params: dict = {"symbol": symbol}
    if date_range:
        start, end = date_range
        sql += " AND date >= :start AND date <= :end"
        params["start"] = str(start)
        params["end"] = str(end)
    sql += " ORDER BY date"
    try:
        df = read_sql(sql, params=params)
        df["date"] = _parse_date(df["date"])
        return df.dropna(subset=["date"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_symbol_earnings(symbol: str, limit: int = 8) -> pd.DataFrame:
    """Prochains résultats publiés pour un symbole."""
    if not pg_enabled() or not symbol:
        return pd.DataFrame()
    try:
        return read_sql(
            """
            SELECT date, "when" AS moment
            FROM earnings.earnings_calendar
            WHERE act_symbol = :symbol
            ORDER BY date DESC
            LIMIT :limit
            """,
            params={"symbol": symbol.upper(), "limit": limit},
        )
    except Exception:
        return pd.DataFrame()


def _date_bounds(symbol: str) -> tuple[date, date] | None:
    bounds: list[tuple[date, date]] = []
    ohlcv = load_stock_ohlcv(symbol)
    if not ohlcv.empty:
        bounds.append((ohlcv["date"].min().date(), ohlcv["date"].max().date()))
    if _lakehouse_matches_symbol(symbol):
        gold = load_gold()
        if not gold.empty:
            bounds.append((gold["date"].min().date(), gold["date"].max().date()))
    if not bounds:
        return None
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


@st.cache_data(ttl=120)
def load_data_overview(symbol: str = DEFAULT_SYMBOL) -> list[dict]:
    """Statistiques par source de données (filtrées par symbole)."""
    rows: list[dict] = []
    symbol = _active_symbol(symbol)

    def _lakehouse_row(label: str, path: str, date_col: str, unit: str) -> None:
        if not exists(path):
            return
        try:
            r = query_duckdb(
                f"""
                SELECT
                    MIN(CAST("{date_col}" AS DATE)) AS dmin,
                    MAX(CAST("{date_col}" AS DATE)) AS dmax,
                    COUNT(*) AS n
                FROM data_table
                """,
                {"data_table": path},
            ).iloc[0]
            rows.append({
                "source": label,
                "debut": pd.to_datetime(r["dmin"]),
                "fin": pd.to_datetime(r["dmax"]),
                "volume": int(r["n"]),
                "unite": unit,
            })
        except Exception:
            pass

    def _reddit_row(sym: str | None = None) -> None:
        path = silver_data_path("news_reddit")
        if not exists(path):
            path = bronze_data_path("news_reddit")
        if not exists(path):
            return
        sym = _active_symbol(sym) if sym else None
        sym_filter = ""
        label = "Messages Reddit"
        if sym and not _lakehouse_matches_symbol(sym):
            sym_filter = f" AND UPPER(News) LIKE '%{sym}%'"
            label = f"Messages Reddit (mentionnant {sym})"
        try:
            r = query_duckdb(
                f"""
                SELECT
                    MIN(CAST("Date" AS DATE)) AS dmin,
                    MAX(CAST("Date" AS DATE)) AS dmax,
                    COUNT(*) AS n
                FROM data_table
                WHERE News IS NOT NULL AND TRIM(CAST(News AS VARCHAR)) != ''
                {sym_filter}
                """,
                {"data_table": path},
            ).iloc[0]
            if int(r["n"] or 0) > 0:
                rows.append({
                    "source": label,
                    "debut": pd.to_datetime(r["dmin"]),
                    "fin": pd.to_datetime(r["dmax"]),
                    "volume": int(r["n"]),
                    "unite": "messages",
                })
        except Exception:
            pass

    if pg_enabled():
        try:
            r = read_sql(
                """
                SELECT MIN(date) AS dmin, MAX(date) AS dmax, COUNT(*) AS n
                FROM stocks.ohlcv
                WHERE act_symbol = :symbol
                """,
                params={"symbol": symbol},
            ).iloc[0]
            if int(r["n"] or 0) > 0:
                rows.append({
                    "source": f"{symbol} (historique complet)",
                    "debut": pd.to_datetime(r["dmin"]),
                    "fin": pd.to_datetime(r["dmax"]),
                    "volume": int(r["n"]),
                    "unite": "jours de bourse",
                })
        except Exception:
            pass

        try:
            opt_n = _symbol_row_count("options", "option_chain", symbol)
            if opt_n > 0:
                rows.append({
                    "source": f"Options ({symbol})",
                    "debut": pd.NaT,
                    "fin": pd.NaT,
                    "volume": opt_n,
                    "unite": "contrats (dernière date)",
                })
        except Exception:
            pass

        try:
            earn_n = _symbol_row_count("earnings", "earnings_calendar", symbol)
            if earn_n > 0:
                r = read_sql(
                    """
                    SELECT MIN(date) AS dmin, MAX(date) AS dmax
                    FROM earnings.earnings_calendar
                    WHERE act_symbol = :symbol
                    """,
                    params={"symbol": symbol},
                ).iloc[0]
                rows.append({
                    "source": f"Résultats ({symbol})",
                    "debut": pd.to_datetime(r["dmin"]),
                    "fin": pd.to_datetime(r["dmax"]),
                    "volume": earn_n,
                    "unite": "publications",
                })
        except Exception:
            pass

    if _lakehouse_matches_symbol(symbol):
        _lakehouse_row(
            f"{symbol} (données récentes)",
            bronze_data_path("stock_prices"),
            "Date",
            "jours de bourse",
        )
        _reddit_row()
        _lakehouse_row(
            f"Jours analysés ({symbol} + Reddit)",
            gold_data_path(),
            "date",
            "jours de bourse",
        )
    else:
        _reddit_row(symbol)

    return rows


@st.cache_data(ttl=300)
def load_warehouse_stats(symbol: str | None = None) -> pd.DataFrame:
    if not pg_enabled():
        return pd.DataFrame()
    try:
        base = warehouse_table_stats()
        if not symbol:
            return base
        symbol = _active_symbol(symbol)
        filtered_rows: list[dict] = []
        for row in base.itertuples(index=False):
            schema, table, total = row.schema, row.table_name, int(row.rows or 0)
            if schema in SYMBOL_SCOPED_TABLES and table in SYMBOL_SCOPED_TABLES[schema]:
                count = _symbol_row_count(schema, table, symbol)
            elif schema == "stocks" and table == "symbol":
                count = _symbol_row_count("stocks", "symbol", symbol)
            else:
                count = total
            filtered_rows.append({
                "schema": schema,
                "table_name": table,
                "rows": count,
            })
        return pd.DataFrame(filtered_rows)
    except Exception:
        return pd.DataFrame()


def render_warehouse_stats(
    stats: pd.DataFrame,
    category_filter: str | None = None,
    symbol: str | None = None,
) -> None:
    symbol = _active_symbol(symbol)
    symbol_name = load_symbol_name(symbol)
    st.markdown("#### Données financières structurées")
    st.caption(f"Filtre actif : **{symbol}** — {symbol_name}")
    if stats.empty:
        st.info(
            "Les données financières (prix, options, taux, résultats) "
            "ne sont pas encore disponibles dans la base."
        )
        return

    stats = stats.copy()
    stats["rows"] = stats["rows"].fillna(0).astype(int)
    stats["label"] = stats["schema"].map(lambda s: DOLT_SCHEMA_LABELS.get(s, s))
    if category_filter and category_filter != "Toutes":
        inv = {v: k for k, v in DOLT_SCHEMA_LABELS.items()}
        schema_key = inv.get(category_filter, category_filter)
        stats = stats[stats["schema"] == schema_key]
        if stats.empty:
            st.warning("Aucune donnée pour cette catégorie.")
            return

    schema_totals = stats.groupby(["schema", "label"], as_index=False)["rows"].sum()

    col_chart, col_cards = st.columns([1, 1.4])
    with col_chart:
        donut = (
            alt.Chart(schema_totals)
            .mark_arc(innerRadius=55, outerRadius=95, padAngle=0.02)
            .encode(
                theta=alt.Theta("rows:Q", stack=True),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        range=["#6366f1", "#f97316", "#22c55e", "#ec4899"]
                    ),
                    legend=alt.Legend(title="Catégories", orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Catégorie"),
                    alt.Tooltip("rows:Q", title="Enregistrements", format=","),
                ],
            )
            .properties(height=280, title="Répartition par catégorie")
        )
        _show_altair(donut)

    with col_cards:
        n = len(schema_totals)
        card_cols = st.columns(min(n, 2))
        for i, row in enumerate(schema_totals.itertuples(index=False)):
            with card_cols[i % len(card_cols)]:
                desc = DOLT_SCHEMA_DESCRIPTIONS.get(row.schema, "")
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-header">
                            <span class="badge badge-volume">{row.label}</span>
                        </div>
                        <div class="metric-value">{_fmt_compact(int(row.rows))}</div>
                        <div class="metric-label">enregistrements</div>
                        <div class="metric-sub">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    grand_total = int(stats["rows"].sum())
    st.caption(
        f"**{_fmt_compact(grand_total)}** enregistrements · "
        f"**{len(stats)}** jeux de données"
    )

    detail = stats.assign(
        Catégorie=stats["schema"].map(lambda s: DOLT_SCHEMA_LABELS.get(s, s)),
        Contenu=stats["table_name"].map(dolt_table_label),
        Volume=stats["rows"],
    )
    bar = (
        alt.Chart(detail)
        .mark_bar(color="#818cf8")
        .encode(
            y=alt.Y(
                "Contenu:N",
                sort=alt.EncodingSortField(field="Volume", order="descending"),
                title="",
            ),
            x=alt.X("Volume:Q", title="Enregistrements", axis=alt.Axis(format=",d")),
            tooltip=[
                alt.Tooltip("Catégorie:N", title="Catégorie"),
                alt.Tooltip("Contenu:N", title="Contenu"),
                alt.Tooltip("Volume:Q", title="Volume", format=","),
            ],
        )
        .properties(height=max(280, len(detail) * 28), title=f"Détail par jeu de données — {symbol}")
    )
    _show_altair(bar)
    if "option_chain" in stats["table_name"].tolist():
        st.caption(
            "Les options affichent le nombre de contrats à la **dernière date disponible** "
            f"pour {symbol} (requête optimisée)."
        )


def _fmt_period_fr(debut: pd.Timestamp, fin: pd.Timestamp) -> str:
    if pd.isna(debut) or pd.isna(fin):
        return "—"
    return f"{debut.strftime('%d/%m/%Y')} → {fin.strftime('%d/%m/%Y')}"


def _years_span(debut: pd.Timestamp, fin: pd.Timestamp) -> str:
    if pd.isna(debut) or pd.isna(fin):
        return "—"
    days = (fin - debut).days
    years = days / 365.25
    if years >= 1.5:
        return f"{years:.0f} ans"
    if years >= 1:
        return "1 an"
    return f"{days} jours"


def _pick(df: pd.DataFrame, name: str) -> float:
    match = df[df["feature"] == name]
    return float(match["importance"].iloc[0]) if not match.empty else 0.0


SOURCE_CHART_LABELS = {
    "Dow Jones (historique complet)": "Dow Jones — historique",
    "Dow Jones (données récentes)": "Dow Jones — récent",
    "Messages Reddit": "Messages Reddit",
    "Jours analysés (Dow Jones + Reddit)": "Jours croisés",
}


def _overview_chart_label(source: str) -> str:
    if source in SOURCE_CHART_LABELS:
        return SOURCE_CHART_LABELS[source]
    if source.endswith(" (historique complet)"):
        return source.replace(" (historique complet)", " — historique")
    return source


# ---------------------------------------------------------------------------
# Render sections
# ---------------------------------------------------------------------------


def _fmt_last_update(report: dict | None) -> str:
    if not report or not report.get("generated_at"):
        return "—"
    try:
        dt = pd.to_datetime(report["generated_at"])
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return str(report.get("generated_at", ""))[:10]


def render_header(
    report: dict | None,
    overview: list[dict],
    ml_metrics: dict | None,
    symbol: str,
    symbol_name: str,
) -> None:
    last_update = _fmt_last_update(report)
    accuracy = f"{ml_metrics.get('accuracy', 0):.0%}" if ml_metrics else "—"
    reddit_row = next((r for r in overview if "Reddit" in r["source"]), None)
    reddit_n = _fmt_compact(reddit_row["volume"]) if reddit_row else "—"
    if _lakehouse_matches_symbol(symbol):
        subtitle = (
            f"Croisez l'évolution de {symbol} ({symbol_name}) et les discussions Reddit "
            "pour estimer si le marché va monter ou baisser demain."
        )
    else:
        subtitle = (
            f"Explorez l'historique de {symbol} ({symbol_name}), ses options, résultats "
            "et les discussions Reddit qui le mentionnent."
        )
    st.markdown(
        f"""
        <div class="hero">
            <h1>Prévisions du marché à partir des actualités</h1>
            <p>{subtitle}</p>
            <div class="hero-pills">
                <span class="hero-pill">📈 {symbol} — {symbol_name}</span>
                <span class="hero-pill">📅 Mise à jour {last_update}</span>
                <span class="hero-pill">💬 {reddit_n} messages Reddit</span>
                <span class="hero-pill">✅ {accuracy} prévisions correctes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _flow_node(icon: str, name: str, sub: str, ok: bool) -> str:
    cls = "flow-node done" if ok else "flow-node wait"
    return (
        f'<div class="{cls}">'
        f'<div class="icon">{icon}</div>'
        f'<div class="name">{name}</div>'
        f'<div class="sub">{sub}</div>'
        f"</div>"
    )


def render_pipeline_flow(report: dict | None, ml_ok: bool, symbol: str, symbol_name: str) -> None:
    st.caption(f"De la collecte des informations à la prévision — symbole **{symbol}**.")

    bronze_ok = _layer_ok(report, "bronze")
    silver_ok = _layer_ok(report, "silver")
    gold_ok = _layer_ok(report, "gold") or exists(gold_data_path())
    ingest_ok = bronze_ok
    stock_label = symbol if len(symbol) <= 8 else symbol[:8] + "…"
    stock_sub = symbol_name[:28] + ("…" if len(symbol_name) > 28 else "")

    with st.container(border=True):
        st.markdown("**⬡ De la collecte à la prévision**")

        src1, src2 = st.columns(2)
        with src1:
            st.markdown(
                _flow_node("📈", stock_label, stock_sub or "Prix historiques", bronze_ok),
                unsafe_allow_html=True,
            )
        with src2:
            reddit_sub = (
                f"Mentionnant {symbol}" if not _lakehouse_matches_symbol(symbol) else "Titres et discussions"
            )
            st.markdown(_flow_node("💬", "Reddit", reddit_sub, bronze_ok), unsafe_allow_html=True)

        st.markdown('<div class="flow-arrow-v">↓</div>', unsafe_allow_html=True)

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            ingest_cls = "flow-ingest" if ingest_ok else "flow-node wait"
            st.markdown(
                f'<div class="{ingest_cls}">'
                f'<div class="icon">⚙️</div>'
                f'<div class="name">Collecte automatique</div>'
                f'<div class="sub">Récupération des données</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="flow-arrow-v">↓</div>', unsafe_allow_html=True)

        steps = [
            ("📥", "Données brutes", "Telles que reçues", bronze_ok),
            ("✨", "Nettoyage", "Vérification et tri", silver_ok),
            ("📊", "Indicateurs", "Chiffres par jour", gold_ok),
            ("🔮", "Prévision", "Hausse ou baisse ?", ml_ok),
        ]
        cols = st.columns([3, 0.4, 3, 0.4, 3, 0.4, 3])
        for i, (icon, name, sub, ok) in enumerate(steps):
            col_idx = i * 2
            with cols[col_idx]:
                st.markdown(_flow_node(icon, name, sub, ok), unsafe_allow_html=True)
            if i < len(steps) - 1:
                with cols[col_idx + 1]:
                    st.markdown('<div class="flow-arrow-h">→</div>', unsafe_allow_html=True)


def render_kpi_cards(
    ml_metrics: dict | None,
    overview: list[dict],
    symbol: str,
) -> None:
    symbol = _active_symbol(symbol)
    symbol_name = load_symbol_name(symbol)
    st.caption(f"Vue d'ensemble des données pour **{symbol}** ({symbol_name}).")

    accuracy = ml_metrics.get("accuracy", 0) if ml_metrics else 0

    pg_row = next((r for r in overview if "historique complet" in r["source"]), None)
    reddit_row = next((r for r in overview if "Reddit" in r["source"]), None)
    combined_row = next((r for r in overview if "analysés" in r["source"]), None)
    options_row = next((r for r in overview if r["source"].startswith("Options")), None)
    earnings_row = next((r for r in overview if r["source"].startswith("Résultats")), None)

    hist_value = _years_span(pg_row["debut"], pg_row["fin"]) if pg_row else "—"
    reddit_value = _fmt_compact(reddit_row["volume"]) if reddit_row else "—"

    if combined_row:
        combined_value = f"{combined_row['volume']:,}".replace(",", " ")
        analysis_label = f"jours croisés ({symbol})"
        analysis_sub = f"{symbol} + Reddit"
    elif options_row or earnings_row:
        opt_n = int(options_row["volume"]) if options_row else 0
        earn_n = int(earnings_row["volume"]) if earnings_row else 0
        combined_value = _fmt_compact(opt_n + earn_n) if (opt_n + earn_n) else "—"
        analysis_label = "options & résultats"
        analysis_sub = f"données {symbol}"
    else:
        combined_value = "—"
        analysis_label = "données croisées"
        analysis_sub = symbol

    if ml_metrics:
        ml_value = f"{accuracy:.0%}"
        ml_sub = f"correctes — {symbol}"
    else:
        ml_value = "—"
        ml_sub = "modèle non entraîné"

    cards = [
        ("Historique", "badge-temps", "📅", hist_value, f"de {symbol}", symbol_name[:24]),
        ("Réseaux sociaux", "badge-social", "💬", reddit_value, "messages Reddit", symbol if not _lakehouse_matches_symbol(symbol) else "tous titres"),
        ("Analyse", "badge-volume", "🗄️", combined_value, analysis_label, analysis_sub),
        ("Fiabilité", "badge-ml", "✅", ml_value, "de prévisions", ml_sub),
    ]

    cols = st.columns(4)
    for col, (badge, badge_cls, icon, value, label, sub) in zip(cols, cards, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-header">
                        <span class="badge {badge_cls}">{badge}</span>
                        <span>{icon}</span>
                    </div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if overview:
        st.markdown("#### Détail par source")
        sources = [r["source"] for r in overview]
        selected = st.multiselect(
            "Filtrer les sources",
            sources,
            default=sources,
            label_visibility="collapsed",
        )
        filtered = [r for r in overview if r["source"] in selected]
        vol_df = pd.DataFrame([
            {
                "Source": _overview_chart_label(row["source"]),
                "Volume": int(row["volume"]),
                "unité": row["unite"],
            }
            for row in filtered
            if int(row.get("volume") or 0) > 0
        ])
        if vol_df.empty:
            st.info("Aucun volume à afficher pour les sources sélectionnées.")
        else:
            units = vol_df["unité"].unique().tolist()
            if len(units) > 1:
                st.caption(
                    "Les sources n'utilisent pas la même unité "
                    f"({', '.join(units)}) : chaque groupe est affiché séparément."
                )
            for unit in units:
                unit_df = vol_df[vol_df["unité"] == unit].copy()
                vol_chart = (
                    alt.Chart(unit_df)
                    .mark_bar(color="#6366f1")
                    .encode(
                        y=alt.Y(
                            "Source:N",
                            sort=alt.EncodingSortField(field="Volume", order="descending"),
                            title="",
                        ),
                        x=alt.X(
                            "Volume:Q",
                            title=f"Volume ({unit})",
                            axis=alt.Axis(format=",d"),
                            scale=alt.Scale(domain=[0, unit_df["Volume"].max() * 1.08]),
                        ),
                        tooltip=[
                            alt.Tooltip("Source:N", title="Source"),
                            alt.Tooltip("Volume:Q", title="Volume", format=","),
                            alt.Tooltip("unité:N", title="Unité"),
                        ],
                    )
                    .properties(height=max(160, len(unit_df) * 52))
                )
                _show_altair(vol_chart)

        with st.expander("Voir le tableau détaillé", expanded=False):
            table = pd.DataFrame([
                {
                    "Source": row["source"],
                    "Période": _fmt_period_fr(row["debut"], row["fin"]),
                    "Durée": _years_span(row["debut"], row["fin"]),
                    "Volume": f"{row['volume']:,} {row['unite']}".replace(",", " "),
                }
                for row in filtered
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)


def _pick_stock_year(years: list[int], symbol: str) -> int | None:
    """Sélecteur d'année cliquable (pills Streamlit ou boutons de repli)."""
    symbol = symbol.upper()
    state_key = f"stock_price_year_{symbol}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    st.markdown(
        '<p class="year-picker-caption">Cliquez une année pour afficher le détail</p>',
        unsafe_allow_html=True,
    )

    if hasattr(st, "pills"):
        selected = st.pills(
            "Année",
            options=years,
            format_func=str,
            selection_mode="single",
            key=state_key,
            label_visibility="collapsed",
        )
        return int(selected) if selected is not None else None

    cols = st.columns(min(len(years), 10))
    selected: int | None = st.session_state[state_key]
    for idx, year in enumerate(years):
        with cols[idx % len(cols)]:
            btn_type = "primary" if selected == year else "secondary"
            if st.button(str(year), key=f"{state_key}_{year}", type=btn_type):
                st.session_state[state_key] = year
                selected = year
    return selected


def _build_stock_price_figure(
    ohlcv: pd.DataFrame,
    symbol: str,
    highlight_year: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ohlcv["date"],
            y=ohlcv["close"],
            mode="lines",
            name="Clôture",
            line={"color": "#4338ca", "width": 2.5},
            hovertemplate=(
                "Date : %{x|%d/%m/%Y}<br>"
                "Clôture : %{y:,.2f} $<br>"
                "<extra></extra>"
            ),
        )
    )
    if highlight_year is not None:
        fig.add_vrect(
            x0=f"{highlight_year}-01-01",
            x1=f"{highlight_year}-12-31",
            fillcolor="#4338ca",
            opacity=0.1,
            line_width=0,
        )
    fig.update_layout(
        height=340,
        title=f"Cours de clôture — {symbol.upper()}",
        xaxis_title="Date",
        yaxis_title="Clôture (USD)",
        hovermode="x unified",
        margin={"l": 48, "r": 16, "t": 48, "b": 40},
        showlegend=False,
    )
    fig.update_xaxes(dtick="M12", tickformat="%Y")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def _render_year_stock_detail(
    ohlcv: pd.DataFrame,
    year: int,
    symbol: str,
) -> None:
    year_df = ohlcv[ohlcv["date"].dt.year == year].sort_values("date")
    if year_df.empty:
        return

    st.markdown(f"#### Détail {year} — {symbol.upper()}")

    start_close = float(year_df.iloc[0]["close"])
    end_close = float(year_df.iloc[-1]["close"])
    ytd_change = (end_close / start_close - 1) * 100 if start_close else 0
    high = float(year_df["high"].max())
    low = float(year_df["low"].min())
    avg_vol = float(year_df["volume"].mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variation annuelle", f"{ytd_change:+.1f} %")
    m2.metric("Plus haut", f"{high:,.2f} $")
    m3.metric("Plus bas", f"{low:,.2f} $")
    m4.metric("Volume moyen / jour", f"{avg_vol:,.0f}")

    monthly = (
        year_df.set_index("date")
        .resample("ME")
        .agg({"close": "last", "volume": "sum"})
        .reset_index()
    )
    monthly["month"] = monthly["date"].dt.strftime("%b")
    monthly_chart = (
        alt.Chart(monthly)
        .mark_bar(color="#818cf8")
        .encode(
            x=alt.X("month:N", sort=list(monthly["month"]), title="Mois"),
            y=alt.Y("close:Q", title="Clôture fin de mois (USD)"),
            tooltip=[
                alt.Tooltip("month:N", title="Mois"),
                alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=","),
            ],
        )
        .properties(height=220, title=f"Clôture mensuelle — {year}")
    )
    _show_altair(monthly_chart)

    with st.expander(f"Voir les {len(year_df)} séances de {year}", expanded=False):
        view = year_df.copy()
        view["Date"] = view["date"].dt.strftime("%d/%m/%Y")
        view["Var. (%)"] = view["daily_return_pct"].map(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        )
        st.dataframe(
            view[["Date", "open", "high", "low", "close", "volume", "Var. (%)"]].rename(
                columns={
                    "open": "Ouverture",
                    "high": "Plus haut",
                    "low": "Plus bas",
                    "close": "Clôture",
                    "volume": "Volume",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_stock_price(symbol: str, date_range: tuple | None = None) -> None:
    """Graphique des cours pour le symbole sélectionné (PostgreSQL)."""
    if not symbol:
        return

    ohlcv = load_stock_ohlcv(symbol, date_range)
    earnings = load_symbol_earnings(symbol)
    dividends = load_symbol_dividends(symbol)

    st.markdown(
        f"""
        <div class="card">
            <div class="card-header">
                <strong>Évolution du cours — {symbol.upper()}</strong>
                <span class="badge badge-volume">Actions</span>
            </div>
            <p class="card-desc">Historique des prix de clôture depuis la base financière
            (PostgreSQL). Survolez la courbe, puis cliquez une année pour voir le détail.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ohlcv.empty:
        st.info(
            f"Aucune donnée de cours pour **{symbol.upper()}** dans PostgreSQL. "
            "Essayez un autre symbole ou vérifiez l'import Dolt."
        )
        return

    ohlcv = ohlcv.copy()
    ohlcv["daily_return_pct"] = ohlcv["close"].pct_change() * 100

    col_chart, col_stats = st.columns([2.2, 1])
    years = sorted(ohlcv["date"].dt.year.unique().tolist())
    year_state_key = f"stock_price_year_{symbol.upper()}"
    raw_year = st.session_state.get(year_state_key)
    selected_year = int(raw_year) if raw_year is not None else None

    with col_chart:
        st.plotly_chart(
            _build_stock_price_figure(ohlcv, symbol, highlight_year=selected_year),
            use_container_width=True,
        )
        selected_year = _pick_stock_year(years, symbol.upper())
        if selected_year is not None:
            _render_year_stock_detail(ohlcv, selected_year, symbol.upper())

    with col_stats:
        last = ohlcv.iloc[-1]
        first = ohlcv.iloc[0]
        change = (last["close"] / first["close"] - 1) * 100 if first["close"] else 0
        st.metric("Dernier cours", f"{last['close']:,.2f} $")
        st.metric("Variation période", f"{change:+.1f} %")
        st.metric("Jours de cotation", f"{len(ohlcv):,}".replace(",", " "))
        if not earnings.empty:
            st.markdown("**Publications récentes**")
            earn_view = earnings.copy()
            earn_view["date"] = pd.to_datetime(earn_view["date"], errors="coerce")
            earn_view = earn_view.dropna(subset=["date"])
            earn_view["Date"] = earn_view["date"].dt.strftime("%d/%m/%Y")
            st.dataframe(
                earn_view[["Date", "moment"]].rename(columns={"moment": "Moment"}),
                use_container_width=True,
                hide_index=True,
            )
        if not dividends.empty:
            st.markdown("**Dividendes récents**")
            div_view = dividends.copy()
            div_view["ex_date"] = pd.to_datetime(div_view["ex_date"], errors="coerce")
            div_view = div_view.dropna(subset=["ex_date"])
            div_view["Date"] = div_view["ex_date"].dt.strftime("%d/%m/%Y")
            st.dataframe(
                div_view[["Date", "amount"]].rename(columns={"amount": "Montant ($)"}),
                use_container_width=True,
                hide_index=True,
            )


def _build_market_chart_df(
    symbol: str,
    gold: pd.DataFrame,
    date_range: tuple | None,
) -> tuple[pd.DataFrame, str]:
    """Assemble ambiance Reddit + rendement du symbole pour le graphique combiné."""
    symbol = _active_symbol(symbol)

    if _lakehouse_matches_symbol(symbol) and not gold.empty:
        chart_df = gold.copy()
        if date_range:
            start, end = date_range
            chart_df = chart_df[
                (chart_df["date"] >= pd.Timestamp(start))
                & (chart_df["date"] <= pd.Timestamp(end))
            ]
        if chart_df.empty:
            return pd.DataFrame(), symbol
        chart_df["sentiment"] = chart_df["finance_news_ratio"].fillna(0)
        if chart_df["sentiment"].max() > 1:
            chart_df["sentiment"] = chart_df["sentiment"] / chart_df["sentiment"].max()
        chart_df["sentiment_ma"] = chart_df["sentiment"].rolling(5, min_periods=1).mean()
        chart_df["market_return"] = chart_df["daily_return_pct"].fillna(0)
        return chart_df, symbol

    ohlcv = load_stock_ohlcv(symbol, date_range)
    if ohlcv.empty:
        return pd.DataFrame(), symbol

    chart_df = ohlcv[["date", "close"]].copy()
    chart_df["market_return"] = chart_df["close"].pct_change() * 100
    sentiment = load_symbol_sentiment_series(symbol, date_range)
    if not sentiment.empty:
        chart_df = chart_df.merge(sentiment[["date", "sentiment"]], on="date", how="left")
        chart_df["sentiment"] = chart_df["sentiment"].fillna(0)
        chart_df["sentiment_ma"] = chart_df["sentiment"].rolling(5, min_periods=1).mean()
    else:
        chart_df["sentiment_ma"] = 0.0
    return chart_df, symbol


def render_sentiment_vs_market(
    symbol: str,
    gold: pd.DataFrame,
    date_range: tuple | None = None,
) -> None:
    symbol = _active_symbol(symbol)
    symbol_name = load_symbol_name(symbol)
    chart_df, symbol = _build_market_chart_df(symbol, gold, date_range)

    if chart_df.empty:
        st.warning(
            f"Aucune donnée disponible pour **{symbol}** sur la période sélectionnée. "
            "Essayez d'élargir la période ou choisissez un autre symbole."
        )
        return

    market_label = f"Évolution de {symbol} (%)"
    reddit_note = (
        "Ambiance Reddit (titres généraux)"
        if _lakehouse_matches_symbol(symbol)
        else f"Ambiance Reddit (titres liés à {symbol})"
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-header">
                <strong>Ambiance des actualités vs évolution de {symbol}</strong>
                <span class="badge badge-important">Interactif</span>
            </div>
            <p class="card-desc">Survolez les courbes, zoomez avec la molette ou faites glisser
            pour sélectionner une période. Bleu = {reddit_note} · Orange = {market_label} ({symbol_name}).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plot = chart_df[["date", "sentiment_ma", "market_return"]].melt(
        id_vars="date", var_name="serie", value_name="valeur"
    )
    plot["serie"] = plot["serie"].map({
        "sentiment_ma": reddit_note,
        "market_return": market_label,
    })

    brush = alt.selection_interval(encodings=["x"])
    base = alt.Chart(plot).encode(x="date:T")
    chart = (
        base.mark_line(interpolate="monotone", strokeWidth=2.5)
        .encode(
            y=alt.Y("valeur:Q", title=""),
            color=alt.Color(
                "serie:N",
                scale=alt.Scale(
                    domain=[reddit_note, market_label],
                    range=["#6366f1", "#f97316"],
                ),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            opacity=alt.condition(brush, alt.value(1), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d/%m/%Y"),
                alt.Tooltip("serie:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur", format=".2f"),
            ],
        )
        .add_params(brush)
        .properties(height=360)
    )
    _show_altair(chart)


def render_bottom_charts(
    sentiment_dist: pd.DataFrame,
    features: pd.DataFrame,
    symbol: str,
) -> None:
    symbol = _active_symbol(symbol)
    reddit_scope = (
        "tous les titres Reddit"
        if _lakehouse_matches_symbol(symbol)
        else f"les titres mentionnant {symbol}"
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <strong>Repartition des sentiments</strong>
                    <span class="badge badge-insight">Analyse</span>
                </div>
                <p class="card-desc">Répartition des titres Reddit selon leur ton
                ({reddit_scope}) : optimiste, neutre ou inquiétant.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if sentiment_dist.empty:
            st.info(
                f"Aucun titre Reddit trouvé pour **{symbol}** sur la période. "
                "Essayez un autre symbole ou élargissez la période."
                if not _lakehouse_matches_symbol(symbol)
                else "Les données Reddit ne sont pas encore disponibles."
            )
        else:
            colors = {"Positif": "#22c55e", "Neutre": "#9ca3af", "Négatif": "#ef4444"}
            chart_df = sentiment_dist.copy()
            chart_df["sentiment"] = chart_df["sentiment"].astype(str)
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("sentiment:N", title="", sort=["Positif", "Neutre", "Négatif"]),
                    y=alt.Y("count:Q", title="Nombre de titres"),
                    color=alt.Color(
                        "sentiment:N",
                        scale=alt.Scale(
                            domain=list(colors.keys()),
                            range=list(colors.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("sentiment:N", title="Sentiment"),
                        alt.Tooltip("count:Q", title="Titres", format=","),
                    ],
                )
                .properties(height=280)
            )
            _show_altair(chart)

    with col2:
        ml_note = "."
        st.markdown(
            f"""
            <div class="card">
                <div class="card-header">
                    <strong>Ce qui influence le plus la prévision</strong>
                    <span class="badge badge-social">Prévision — {symbol}</span>
                </div>
                <p class="card-desc">Les éléments qui pèsent le plus dans la décision
                « {symbol} va-t-il monter demain ? »{ml_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if features.empty:
            st.info(
                f"Aucun modèle entraîné pour **{symbol}**. "
                "Lancez `python scripts/train_ml_symbols.py` pour générer les prévisions."
            )
        else:
            chart_df = features.copy()
            max_imp = chart_df["importance"].max() or 1
            if "pct" not in chart_df.columns:
                chart_df["pct"] = (chart_df["importance"] / max_imp * 100).round(0)
            chart_df = chart_df.sort_values("pct", ascending=True)

            influence_chart = (
                alt.Chart(chart_df)
                .mark_bar(color="#fdba74")
                .encode(
                    y=alt.Y(
                        "feature:N",
                        title="",
                        sort=alt.EncodingSortField(field="pct", order="ascending"),
                    ),
                    x=alt.X("pct:Q", title="Importance relative (%)"),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Facteur"),
                        alt.Tooltip("pct:Q", title="Importance (%)", format=".0f"),
                    ],
                )
                .properties(height=260)
            )
            _show_altair(influence_chart)


def render_ml_summary(
    ml_metrics: dict | None,
    predictions: pd.DataFrame,
    symbol: str,
    date_range: tuple | None = None,
) -> None:
    symbol = _active_symbol(symbol)
    symbol_name = load_symbol_name(symbol)

    if not ml_metrics:
        st.info(
            f"Aucun modèle de prévision entraîné pour **{symbol}** ({symbol_name}). "
            "Exécutez `python scripts/train_ml_symbols.py` pour entraîner tous les symboles."
        )
        render_stock_price(symbol, date_range=date_range)
        return

    st.markdown(f"#### Performance des prévisions — {symbol}")
    st.caption(
        f"Modèle entraîné sur l'historique **{symbol}** "
        f"+ discussions Reddit (sentiment marché général)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prévisions correctes", f"{ml_metrics.get('accuracy', 0):.0%}",
              help="Part des jours où la direction (hausse ou baisse) a été bien devinée.")
    c2.metric("Fiabilité des hausses", f"{ml_metrics.get('precision', 0):.0%}",
              help="Quand le modèle annonce une hausse, à quelle fréquence a-t-il raison ?")
    c3.metric("Hausses détectées", f"{ml_metrics.get('recall', 0):.0%}",
              help="Parmi les vraies hausses, combien le modèle a-t-il repérées ?")
    c4.metric("Score global", f"{ml_metrics.get('f1', 0):.0%}",
              help="Synthèse entre fiabilité et couverture des hausses.")
    c5.metric("Jours testés", f"{ml_metrics.get('samples_predict', 0):,}",
              help="Nombre de jours utilisés pour vérifier les prévisions en 2026.")

    if not predictions.empty and "probability_up" in predictions.columns:
        threshold = st.slider(
            "Seuil de confiance « hausse » (%)",
            min_value=40,
            max_value=80,
            value=50,
            step=5,
            help="Ajustez le seuil au-dessus duquel le modèle annonce une hausse.",
        )
        pred_view = predictions.copy()
        pred_view["signal"] = (
            pred_view["probability_up"] >= threshold / 100
        ).map({True: "Hausse prédite", False: "Baisse / stable"})

        col_a, col_b = st.columns([2, 1])
        with col_a:
            pred_chart = (
                alt.Chart(pred_view)
                .mark_area(
                    line={"color": "#6366f1"},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="#c7d2fe", offset=0),
                            alt.GradientStop(color="white", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                    interpolate="monotone",
                )
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("probability_up:Q", title="Probabilité de hausse", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date", format="%d/%m/%Y"),
                        alt.Tooltip("probability_up:Q", title="Probabilité", format=".0%"),
                        alt.Tooltip("signal:N", title="Signal"),
                    ],
                )
                .properties(height=280, title="Confiance du modèle dans le temps")
                .interactive()
            )
            rule = (
                alt.Chart(pd.DataFrame({"y": [threshold / 100]}))
                .mark_rule(color="#ef4444", strokeDash=[6, 4])
                .encode(y="y:Q")
            )
            _show_altair(pred_chart + rule)

        with col_b:
            signal_counts = pred_view["signal"].value_counts().reset_index()
            signal_counts.columns = ["signal", "count"]
            pie = (
                alt.Chart(signal_counts)
                .mark_arc(innerRadius=40)
                .encode(
                    theta="count:Q",
                    color=alt.Color("signal:N", legend=alt.Legend(title="")),
                    tooltip=[alt.Tooltip("signal:N"), alt.Tooltip("count:Q", format=",")],
                )
                .properties(height=280, title=f"Signaux (seuil {threshold}%)")
            )
            _show_altair(pie)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Explorez les onglets pour parcourir les données, les analyses et les prévisions.")
    st.divider()
    if st.button("Actualiser les données", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Recharge les derniers chiffres depuis le lakehouse et PostgreSQL.")

    st.divider()
    st.markdown("**Filtres**")

    default_idx = (
        POPULAR_SYMBOLS.index(DEFAULT_SYMBOL)
        if DEFAULT_SYMBOL in POPULAR_SYMBOLS
        else 0
    )
    symbol_pick = st.selectbox(
        "Symbole boursier",
        [*POPULAR_SYMBOLS, "Autre…"],
        index=default_idx,
        help="Filtre cours, options, résultats, Reddit et graphiques pour ce symbole.",
    )
    if symbol_pick == "Autre…":
        selected_symbol = st.text_input(
            "Saisir un symbole",
            value="",
            placeholder="ex. NFLX",
        ).strip().upper()
    else:
        selected_symbol = symbol_pick

    active_symbol = _active_symbol(selected_symbol)
    symbol_name = load_symbol_name(active_symbol)
    st.caption(f"Actif : **{active_symbol}** — {symbol_name}")

    bounds = _date_bounds(active_symbol)
    date_range = None
    if bounds:
        min_d, max_d = bounds
        date_range = st.slider(
            "Période d'analyse",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            format="DD/MM/YYYY",
        )

    wh_categories = ["Toutes", *DOLT_SCHEMA_LABELS.values()]
    wh_filter = st.selectbox("Catégorie financière", wh_categories, index=0)

    st.divider()
    _reports = load_all_monitoring_reports()
    _latest = _reports[0] if _reports else None
    bronze_ok = _layer_ok(_latest, "bronze")
    gold_ok = exists(gold_data_path())
    ml_ok = load_ml_metrics(active_symbol) is not None
    st.markdown("**État du pipeline**")
    st.markdown(f"{'🟢' if bronze_ok else '🟡'} Collecte des données")
    st.markdown(f"{'🟢' if gold_ok else '🟡'} Indicateurs journaliers")
    st.markdown(f"{'🟢' if ml_ok else '🟡'} Modèle de prévision ({active_symbol})")

report = load_all_monitoring_reports()
latest_report = report[0] if report else None
gold = load_gold()
ml_metrics = load_ml_metrics(active_symbol)
predictions = load_ml_predictions(active_symbol)
sentiment_dist = load_sentiment_distribution(active_symbol)
features = load_feature_importance(active_symbol)
data_overview = load_data_overview(active_symbol)
warehouse_stats = load_warehouse_stats(active_symbol)

render_header(latest_report, data_overview, ml_metrics, active_symbol, symbol_name)

tab_parcours, tab_chiffres, tab_analyses, tab_previsions = st.tabs(
    [
        "Parcours des données",
        "Chiffres clés",
        "Analyses",
        "Prévisions",
    ]
)

with tab_parcours:
    render_pipeline_flow(latest_report, ml_metrics is not None, active_symbol, symbol_name)

with tab_chiffres:
    render_kpi_cards(ml_metrics, data_overview, active_symbol)
    st.markdown("<br>", unsafe_allow_html=True)
    render_warehouse_stats(warehouse_stats, category_filter=wh_filter, symbol=active_symbol)

with tab_analyses:
    render_market_carpet()
    st.markdown("<br>", unsafe_allow_html=True)
    render_stock_price(active_symbol, date_range=date_range)
    st.markdown("<br>", unsafe_allow_html=True)
    render_sentiment_vs_market(active_symbol, gold, date_range=date_range)
    st.markdown("<br>", unsafe_allow_html=True)
    render_bottom_charts(sentiment_dist, features, active_symbol)

with tab_previsions:
    render_ml_summary(ml_metrics, predictions, active_symbol, date_range=date_range)
