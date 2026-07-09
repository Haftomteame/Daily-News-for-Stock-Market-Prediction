#!/usr/bin/env python3
"""Dashboard — Prévisions bourse & actualités (interface grand public)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    ML_PREDICTIONS_PATH,
    bronze_data_path,
    gold_data_path,
    silver_data_path,
)
from src.env import load_dotenv  # noqa: E402
from src.db.postgres import (  # noqa: E402
    DOLT_SCHEMA_DESCRIPTIONS,
    DOLT_SCHEMA_LABELS,
    dolt_table_label,
    pg_enabled,
    read_sql,
    warehouse_table_stats,
)
from src.storage.io import (  # noqa: E402
    exists,
    glob_paths,
    monitoring_path,
    query_duckdb,
    read_json,
    read_joblib,
    read_parquet,
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
    [data-testid="stSidebar"] * { color: #e0e7ff !important; }
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
def load_ml_metrics() -> dict | None:
    return _read_json_local_or_storage(ML_METRICS_PATH, "lakehouse/ml/metrics.json")


@st.cache_data(ttl=60)
def load_ml_predictions() -> pd.DataFrame:
    if not exists(ML_PREDICTIONS_PATH):
        return pd.DataFrame()
    try:
        df = read_parquet(ML_PREDICTIONS_PATH)
        if "date" in df.columns:
            df["date"] = _parse_date(df["date"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sentiment_distribution() -> pd.DataFrame:
    path = silver_data_path("news_reddit")
    if not exists(path):
        path = bronze_data_path("news_reddit")
    if not exists(path):
        return pd.DataFrame()

    try:
        df = query_duckdb(
            """
            SELECT News FROM data_table
            WHERE News IS NOT NULL AND TRIM(CAST(News AS VARCHAR)) != ''
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
    counts["sentiment"] = pd.Categorical(counts["sentiment"], categories=order, ordered=True)
    return counts.sort_values("sentiment")


@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame:
    if not exists(ML_MODEL_PATH):
        return pd.DataFrame()
    try:
        model = read_joblib(ML_MODEL_PATH)
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


@st.cache_data(ttl=120)
def load_data_overview() -> list[dict]:
    """Statistiques par source de données (toutes les données disponibles)."""
    rows: list[dict] = []

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

    if pg_enabled():
        try:
            r = read_sql(
                """
                SELECT MIN(date) AS dmin, MAX(date) AS dmax, COUNT(*) AS n
                FROM stocks.ohlcv
                WHERE act_symbol = 'DIA'
                """,
            ).iloc[0]
            rows.append({
                "source": "Cours Dow Jones (historique complet)",
                "debut": pd.to_datetime(r["dmin"]),
                "fin": pd.to_datetime(r["dmax"]),
                "volume": int(r["n"]),
                "unite": "jours de bourse",
            })
        except Exception:
            pass

    _lakehouse_row(
        "Cours Dow Jones (analyse actuelle)",
        bronze_data_path("stock_prices"),
        "Date",
        "jours de bourse",
    )
    _lakehouse_row(
        "Messages Reddit",
        bronze_data_path("news_reddit"),
        "Date",
        "messages",
    )
    _lakehouse_row(
        "Jours analysés (cours + Reddit)",
        gold_data_path(),
        "date",
        "jours de bourse",
    )
    return rows


@st.cache_data(ttl=300)
def load_warehouse_stats() -> pd.DataFrame:
    if not pg_enabled():
        return pd.DataFrame()
    try:
        return warehouse_table_stats()
    except Exception:
        return pd.DataFrame()


def render_warehouse_stats(stats: pd.DataFrame, category_filter: str | None = None) -> None:
    st.markdown("#### Données financières structurées")
    if stats.empty:
        st.info(
            "Les données financières (cours, options, taux, résultats) "
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
            .interactive()
        )
        st.altair_chart(donut, use_container_width=True)

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
        .mark_bar(cornerRadiusEnd=4, color="#818cf8")
        .encode(
            y=alt.Y("Contenu:N", sort="-x", title=""),
            x=alt.X("Volume:Q", title="Enregistrements"),
            color=alt.Color("Catégorie:N", legend=None),
            tooltip=[
                alt.Tooltip("Catégorie:N", title="Catégorie"),
                alt.Tooltip("Contenu:N", title="Contenu"),
                alt.Tooltip("Volume:Q", title="Volume", format=","),
            ],
        )
        .properties(height=max(220, len(detail) * 22), title="Détail par jeu de données")
        .interactive()
    )
    st.altair_chart(bar, use_container_width=True)


def _fmt_period_fr(debut: pd.Timestamp, fin: pd.Timestamp) -> str:
    return f"{debut.strftime('%d/%m/%Y')} → {fin.strftime('%d/%m/%Y')}"


def _years_span(debut: pd.Timestamp, fin: pd.Timestamp) -> str:
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


def render_header(report: dict | None, overview: list[dict], ml_metrics: dict | None) -> None:
    last_update = _fmt_last_update(report)
    accuracy = f"{ml_metrics.get('accuracy', 0):.0%}" if ml_metrics else "—"
    reddit_row = next((r for r in overview if r["source"] == "Messages Reddit"), None)
    reddit_n = _fmt_compact(reddit_row["volume"]) if reddit_row else "—"
    st.markdown(
        f"""
        <div class="hero">
            <h1>Prévisions du marché à partir des actualités</h1>
            <p>Croisez l'évolution du Dow Jones et les discussions Reddit pour estimer
            si le marché va monter ou baisser demain.</p>
            <div class="hero-pills">
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


def render_pipeline_flow(report: dict | None, ml_ok: bool) -> None:
    st.caption("De la collecte des informations à la prévision du marché.")

    bronze_ok = _layer_ok(report, "bronze")
    silver_ok = _layer_ok(report, "silver")
    gold_ok = _layer_ok(report, "gold") or exists(gold_data_path())
    ingest_ok = bronze_ok

    with st.container(border=True):
        st.markdown("**⬡ De la collecte à la prévision**")

        src1, src2 = st.columns(2)
        with src1:
            st.markdown(_flow_node("📈", "Cours de bourse", "Historique Dow Jones", bronze_ok), unsafe_allow_html=True)
        with src2:
            st.markdown(_flow_node("💬", "Reddit", "Titres et discussions", bronze_ok), unsafe_allow_html=True)

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
) -> None:
    st.caption("Vue d'ensemble de toutes les données disponibles.")

    accuracy = ml_metrics.get("accuracy", 0) if ml_metrics else 0

    pg_row = next((r for r in overview if "historique complet" in r["source"]), None)
    reddit_row = next((r for r in overview if r["source"] == "Messages Reddit"), None)
    combined_row = next((r for r in overview if "analysés" in r["source"]), None)

    hist_value = _years_span(pg_row["debut"], pg_row["fin"]) if pg_row else "—"
    reddit_value = _fmt_compact(reddit_row["volume"]) if reddit_row else "—"
    combined_value = f"{combined_row['volume']:,}".replace(",", " ") if combined_row else "—"

    cards = [
        ("Historique", "badge-temps", "📅", hist_value, "de cours Dow Jones", "depuis 2011"),
        ("Réseaux sociaux", "badge-social", "💬", reddit_value, "messages Reddit", "depuis 2023"),
        ("Analyse", "badge-volume", "🗄️", combined_value, "jours croisés", "cours + Reddit"),
        ("Fiabilité", "badge-ml", "✅", f"{accuracy:.0%}", "de prévisions", "correctes sur 2026"),
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
                "Source": row["source"],
                "Volume": row["volume"],
                "unité": row["unite"],
            }
            for row in filtered
        ])
        if not vol_df.empty:
            vol_chart = (
                alt.Chart(vol_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Source:N", sort="-y", title="", axis=alt.Axis(labelAngle=-25)),
                    y=alt.Y("Volume:Q", title="Volume"),
                    color=alt.Color("Source:N", legend=None, scale=alt.Scale(scheme="category10")),
                    tooltip=[
                        alt.Tooltip("Source:N", title="Source"),
                        alt.Tooltip("Volume:Q", title="Volume", format=","),
                        alt.Tooltip("unité:N", title="Unité"),
                    ],
                )
                .properties(height=280)
                .interactive()
            )
            st.altair_chart(vol_chart, use_container_width=True)

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


def render_sentiment_vs_djia(gold: pd.DataFrame, date_range: tuple | None = None) -> None:
    if gold.empty:
        st.warning(
            "Les graphiques ne sont pas encore disponibles. "
            "Les données seront affichées après la prochaine mise à jour automatique."
        )
        return

    chart_df = gold.copy()
    if date_range:
        start, end = date_range
        chart_df = chart_df[
            (chart_df["date"] >= pd.Timestamp(start))
            & (chart_df["date"] <= pd.Timestamp(end))
        ]
    if chart_df.empty:
        st.info("Aucune donnée pour la période sélectionnée.")
        return

    chart_df["sentiment"] = chart_df["finance_news_ratio"].fillna(0)
    if chart_df["sentiment"].max() > 1:
        chart_df["sentiment"] = chart_df["sentiment"] / chart_df["sentiment"].max()
    chart_df["sentiment_ma"] = chart_df["sentiment"].rolling(5, min_periods=1).mean()
    chart_df["djia_return"] = chart_df["daily_return_pct"].fillna(0)

    st.markdown(
        """
        <div class="card">
            <div class="card-header">
                <strong>Ambiance des actualités vs évolution du Dow Jones</strong>
                <span class="badge badge-important">Interactif</span>
            </div>
            <p class="card-desc">Survolez les courbes, zoomez avec la molette ou faites glisser
            pour sélectionner une période. Bleu = ambiance Reddit · Orange = variation DJIA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plot = chart_df[["date", "sentiment_ma", "djia_return"]].melt(
        id_vars="date", var_name="serie", value_name="valeur"
    )
    plot["serie"] = plot["serie"].map({
        "sentiment_ma": "Ambiance des actualités",
        "djia_return": "Évolution du Dow Jones (%)",
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
                    domain=["Ambiance des actualités", "Évolution du Dow Jones (%)"],
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
    st.altair_chart(chart, use_container_width=True)


def render_bottom_charts(sentiment_dist: pd.DataFrame, features: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <strong>Repartition des sentiments</strong>
                    <span class="badge badge-insight">Analyse</span>
                </div>
                <p class="card-desc">Répartition des titres Reddit selon leur ton :
                optimiste, neutre ou inquiétant.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if sentiment_dist.empty:
            st.info("Les données Reddit ne sont pas encore disponibles.")
        else:
            colors = {"Positif": "#22c55e", "Neutre": "#9ca3af", "Négatif": "#ef4444"}
            chart = (
                alt.Chart(sentiment_dist)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
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
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <strong>Ce qui influence le plus la prévision</strong>
                    <span class="badge badge-social">Prévision</span>
                </div>
                <p class="card-desc">Les éléments qui pèsent le plus dans la décision
                « le marché va-t-il monter demain ? ».</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if features.empty:
            st.info("Le modèle de prévision n'est pas encore disponible.")
        else:
            chart_df = features.copy()
            max_imp = chart_df["importance"].max() or 1
            if "pct" not in chart_df.columns:
                chart_df["pct"] = (chart_df["importance"] / max_imp * 100).round(0)
            chart_df = chart_df.sort_values("pct", ascending=True)

            influence_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusEnd=4, color="#fdba74")
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
            st.altair_chart(influence_chart, use_container_width=True)


def render_ml_summary(ml_metrics: dict | None, predictions: pd.DataFrame) -> None:
    if not ml_metrics:
        st.info("Les résultats de prévision ne sont pas encore disponibles.")
        return

    st.markdown("#### Performance des prévisions")

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
            st.altair_chart(pred_chart + rule, use_container_width=True)

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
            st.altair_chart(pie, use_container_width=True)


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

    gold_for_filter = load_gold()
    date_range = None
    if not gold_for_filter.empty:
        min_d = gold_for_filter["date"].min().date()
        max_d = gold_for_filter["date"].max().date()
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
    ml_ok = load_ml_metrics() is not None
    st.markdown("**État du pipeline**")
    st.markdown(f"{'🟢' if bronze_ok else '🟡'} Collecte des données")
    st.markdown(f"{'🟢' if gold_ok else '🟡'} Indicateurs journaliers")
    st.markdown(f"{'🟢' if ml_ok else '🟡'} Modèle de prévision")

report = load_all_monitoring_reports()
latest_report = report[0] if report else None
gold = load_gold()
ml_metrics = load_ml_metrics()
predictions = load_ml_predictions()
sentiment_dist = load_sentiment_distribution()
features = load_feature_importance()
data_overview = load_data_overview()
warehouse_stats = load_warehouse_stats()

render_header(latest_report, data_overview, ml_metrics)

tab_parcours, tab_chiffres, tab_analyses, tab_previsions = st.tabs(
    [
        "Parcours des données",
        "Chiffres clés",
        "Analyses",
        "Prévisions",
    ]
)

with tab_parcours:
    render_pipeline_flow(latest_report, ml_metrics is not None)

with tab_chiffres:
    render_kpi_cards(ml_metrics, data_overview)
    st.markdown("<br>", unsafe_allow_html=True)
    render_warehouse_stats(warehouse_stats, category_filter=wh_filter)

with tab_analyses:
    render_sentiment_vs_djia(gold, date_range=date_range)
    st.markdown("<br>", unsafe_allow_html=True)
    render_bottom_charts(sentiment_dist, features)

with tab_previsions:
    render_ml_summary(ml_metrics, predictions)
