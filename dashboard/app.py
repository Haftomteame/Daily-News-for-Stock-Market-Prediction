#!/usr/bin/env python3
"""Dashboard Streamlit — KPIs Gold, monitoring et ML."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    ML_PREDICTIONS_PATH,
    MONITORING_HISTORY_PATH,
    bronze_data_path,
    gold_data_path,
)
from src.env import load_dotenv  # noqa: E402
from src.storage.io import (  # noqa: E402
    exists,
    modified_time,
    read_json,
    read_parquet,
    storage_label,
)
from src.db.postgres import pg_enabled, read_table  # noqa: E402

load_dotenv()

st.set_page_config(
    page_title="Data Lakehouse — Stock Market",
    page_icon="📊",
    layout="wide",
)

st.title("Data Lakehouse — Daily News / Stock Market")
st.caption(f"Architecture Medallion : Bronze → Silver → Gold + ML | Stockage : {storage_label()}")


@st.cache_data
def load_gold(_gold_mtime: float) -> pd.DataFrame:
    if pg_enabled():
        try:
            return read_table("gold_daily_market_kpis")
        except Exception:
            pass
    path = gold_data_path()
    if not exists(path):
        return pd.DataFrame()
    return read_parquet(path)


@st.cache_data
def load_monitoring_history(_monitoring_mtime: float) -> pd.DataFrame:
    if not exists(MONITORING_HISTORY_PATH):
        return pd.DataFrame()
    return read_parquet(MONITORING_HISTORY_PATH)


@st.cache_data
def load_ml_metrics(_metrics_mtime: float) -> dict | None:
    if not exists(ML_METRICS_PATH):
        return None
    return read_json(ML_METRICS_PATH)


@st.cache_data
def load_predictions(_predictions_mtime: float) -> pd.DataFrame:
    if not exists(ML_PREDICTIONS_PATH):
        return pd.DataFrame()
    return read_parquet(ML_PREDICTIONS_PATH)


@st.cache_data
def load_stock_1m(_bronze_mtime: float) -> pd.DataFrame:
    path = bronze_data_path("stock_prices_1m")
    if not exists(path):
        return pd.DataFrame()
    df = read_parquet(path)
    for col in ("_ingestion_ts", "_source_file", "_batch_id", "_layer"):
        if col in df.columns:
            df = df.drop(columns=col)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", ascending=False)


with st.sidebar:
    st.subheader("Donnees")
    if st.button("Rafraichir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Orchestration")
    airflow_url = os.getenv("AIRFLOW_URL", "http://localhost:8081")
    if hasattr(st, "link_button"):
        st.link_button("Ouvrir Airflow", airflow_url, use_container_width=True)
    else:
        st.markdown(f"[Ouvrir Airflow]({airflow_url})")

gold = load_gold(modified_time(gold_data_path()))
stock_1m = load_stock_1m(modified_time(bronze_data_path("stock_prices_1m")))
monitoring = load_monitoring_history(modified_time(MONITORING_HISTORY_PATH))
ml_metrics = load_ml_metrics(modified_time(ML_METRICS_PATH))
predictions = load_predictions(modified_time(ML_PREDICTIONS_PATH))

if ml_metrics:
    st.sidebar.caption(
        f"Dernier run : {ml_metrics.get('trained_at', 'N/A')[:19]} | "
        f"Prediction {ml_metrics.get('prediction_year', 'N/A')}"
    )

from src.storage.io import is_hdfs, hdfs_web_url  # noqa: E402

if is_hdfs():
    st.sidebar.markdown(f"**HDFS UI** : [{hdfs_web_url()}]({hdfs_web_url()})")
    base = os.getenv("HDFS_BASE_PATH", "/datax")
    st.sidebar.caption(f"Explorer : {hdfs_web_url()}/explorer.html#/{base.strip('/')}/lakehouse")

tab_overview, tab_realtime, tab_gold, tab_monitoring, tab_ml = st.tabs(
    ["Vue d'ensemble", "Temps reel", "Couche Gold", "Monitoring", "ML"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jours Gold", len(gold) if not gold.empty else 0)
    c2.metric("Runs monitorés", monitoring["batch_id"].nunique() if not monitoring.empty else 0)
    c3.metric("Accuracy ML", f"{ml_metrics['accuracy']:.1%}" if ml_metrics else "N/A")
    c4.metric("Modele", "OK" if exists(ML_MODEL_PATH) else "Absent")

    st.subheader("Flux de donnees")
    st.markdown("""
    | Couche | Donnees | Role |
    |--------|---------|------|
    | **Bronze** | DJIA + Reddit + Combined | ELT brut (HDFS ou local) |
    | **Silver** | Nettoyage, metadata, label ML | Qualite |
    | **Gold** | KPIs journaliers (schema fixe) | Analytics |
    | **ML** | LogisticRegression sur features Gold + Combined | Prediction Label |
    """)

    if gold.empty:
        st.warning("Pipeline non execute. Lancez : `python pipeline/run_pipeline.py`")

with tab_realtime:
    st.subheader("DIA — bougies 1 min (Finnhub WebSocket)")
    st.caption(
        "Source : `lakehouse/bronze/stock_prices_1m/` — "
        "lancez `docker compose --profile stream up -d finnhub-stream` ou le script local."
    )
    if stock_1m.empty:
        st.info(
            "Aucune donnee temps reel. Ajoutez FINNHUB_TOKEN dans .env puis demarrez le stream."
        )
    else:
        latest = stock_1m.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dernier close", f"{latest.get('Close', 0):.2f}")
        c2.metric("Volume (1 min)", f"{int(latest.get('Volume', 0)):,}")
        c3.metric("Bougies", len(stock_1m))
        c4.metric(
            "Derniere bougie",
            pd.to_datetime(latest["Date"]).strftime("%Y-%m-%d %H:%M") if "Date" in latest else "N/A",
        )

        chart_df = stock_1m.sort_values("Date").set_index("Date")
        st.line_chart(chart_df["Close"].tail(120))
        st.dataframe(
            stock_1m.head(30),
            use_container_width=True,
        )

with tab_gold:
    if gold.empty:
        st.info("Aucune donnee Gold disponible.")
    else:
        gold["date"] = pd.to_datetime(gold["date"])
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Cours de cloture DJIA")
            st.line_chart(gold.set_index("date")["close"])

        with col_b:
            st.subheader("Volume de news par jour")
            st.bar_chart(gold.set_index("date")["news_count"])

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Rendement journalier (%)")
            st.line_chart(gold.set_index("date")["daily_return_pct"])

        with col_d:
            st.subheader("Direction du marche")
            direction_counts = gold["market_direction"].value_counts()
            st.bar_chart(direction_counts)

        st.subheader("Echantillon KPIs")
        st.dataframe(
            gold[["date", "close", "daily_return_pct", "news_count", "finance_news_ratio", "market_direction"]]
            .sort_values("date", ascending=False)
            .head(20),
            use_container_width=True,
        )

with tab_monitoring:
    if monitoring.empty:
        st.info("Aucun historique monitoring. Executez le pipeline.")
    else:
        latest = monitoring.groupby("layer").last().reset_index()
        c1, c2, c3 = st.columns(3)
        for i, layer in enumerate(["bronze", "silver", "gold"]):
            layer_data = monitoring[monitoring["layer"] == layer]
            if layer_data.empty:
                continue
            with [c1, c2, c3][i]:
                st.subheader(layer.upper())
                st.metric("Latence (ms)", f"{layer_data['latency_ms'].sum():.0f}")
                st.metric("Stockage (MB)", f"{layer_data['storage_mb'].sum():.2f}")
                st.metric("Qualite", f"{layer_data['quality_score'].mean():.1%}")

        st.subheader("Latence par couche (historique)")
        latency_chart = monitoring.groupby(["batch_id", "layer"])["latency_ms"].sum().reset_index()
        st.line_chart(
            latency_chart.pivot(index="batch_id", columns="layer", values="latency_ms").fillna(0)
        )

        st.subheader("Cout estime par couche")
        cost_chart = monitoring.groupby(["batch_id", "layer"])["cost_usd"].sum().reset_index()
        st.bar_chart(
            cost_chart.pivot(index="batch_id", columns="layer", values="cost_usd").fillna(0)
        )

        st.dataframe(monitoring.sort_values("recorded_at", ascending=False), use_container_width=True)

with tab_ml:
    if not ml_metrics:
        st.info("Modele ML non entraine. Executez le pipeline complet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{ml_metrics['accuracy']:.1%}")
        c2.metric("Precision", f"{ml_metrics['precision']:.1%}")
        c3.metric("Recall", f"{ml_metrics['recall']:.1%}")
        c4.metric("F1", f"{ml_metrics['f1']:.1%}")

        st.subheader("Features utilisees")
        st.code("\n".join(ml_metrics.get("features", [])))

        if ml_metrics.get("prediction_year"):
            st.caption(
                f"Entrainement : {ml_metrics.get('train_period', 'N/A')} | "
                f"Prediction {ml_metrics['prediction_year']} : "
                f"{ml_metrics.get('predict_period', 'N/A')}"
            )

        if not predictions.empty:
            st.subheader(f"Predictions {ml_metrics.get('prediction_year', '')} vs Label reel")
            predictions["date"] = pd.to_datetime(predictions["date"])
            sample = predictions.sort_values("date", ascending=False).head(30)
            if "correct" not in sample.columns:
                sample["correct"] = sample["predicted_label"] == sample["label"]
            st.dataframe(
                sample[["date", "label", "predicted_label", "probability_up", "correct"]],
                use_container_width=True,
            )

            accuracy_over_time = (
                predictions.assign(correct=predictions["predicted_label"] == predictions["label"])
                .set_index("date")["correct"]
                .rolling(50, min_periods=10)
                .mean()
            )
            st.subheader("Accuracy glissante (50 jours)")
            st.line_chart(accuracy_over_time)
