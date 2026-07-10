"""Composant Market Carpet (treemap sectorielle) pour le dashboard Streamlit."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.db.postgres import pg_enabled, read_sql
from src.marketcarpet.builder import build_market_carpet_df
from src.marketcarpet.loader import load_ohlcv_panel
from src.marketcarpet.metrics import MEASUREMENT_LABELS, PERIOD_DAYS
from src.marketcarpet.metrics import color_range_for_measurement
from src.marketcarpet.universe import list_universe_groups, load_universe

COLOR_SCHEMES = {
    "Rouge → Vert": "RdYlGn",
    "Bleu → Rouge": "RdBu_r",
    "Violet → Jaune": "PuOr",
}

PERIOD_LABELS = {
    "1D": "Variation 1 jour",
    "5D": "Variation 5 jours",
    "1M": "Variation 1 mois",
    "3M": "Variation 3 mois",
    "6M": "Variation 6 mois",
    "1Y": "Variation 1 an",
}

SIZE_LABELS = {
    "market_cap": "Capitalisation",
    "equal": "Poids égal",
    "price": "Prix",
}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_carpet(
    group_key: str,
    measurement: str,
    period_key: str,
    size_mode: str,
) -> tuple[pd.DataFrame, dict]:
    return build_market_carpet_df(
        group_key,
        measurement=measurement,
        period_key=period_key,
        size_mode=size_mode,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ohlcv_panel(group_key: str) -> pd.DataFrame:
    symbols = load_universe(group_key)["symbol"].tolist()
    return load_ohlcv_panel(symbols)


def _format_metric(value: float, measurement: str) -> str:
    if measurement == "performance":
        return f"{value:+.2f} %"
    if measurement == "rsi":
        return f"{value:.1f}"
    return f"{value:+.0f}"


def _metric_color(value: float, measurement: str) -> str:
    if measurement == "performance":
        if value > 0:
            return "#15803d"
        if value < 0:
            return "#dc2626"
        return "#6b7280"
    if measurement == "rsi":
        if value >= 70:
            return "#dc2626"
        if value <= 30:
            return "#15803d"
        return "#4338ca"
    if value > 0:
        return "#15803d"
    if value < 0:
        return "#dc2626"
    return "#6b7280"


def _color_midpoint(measurement: str, period_key: str) -> float | None:
    if measurement == "performance":
        return 0.0
    if measurement == "rsi" and period_key == "1D":
        return 50.0
    if measurement == "up_down_days":
        return 0.0
    return None


def _build_treemap(
    df: pd.DataFrame,
    *,
    measurement: str,
    period_key: str,
    color_scheme: str,
    show_symbols: bool,
) -> go.Figure:
    cmin, cmax = color_range_for_measurement(measurement, period_key)
    midpoint = _color_midpoint(measurement, period_key)
    plot_df = df.copy()
    plot_df["tile_text"] = plot_df.get(
        "tile_text",
        plot_df["symbol"],
    )
    plot_df["change_abs_label"] = plot_df.get("change_abs", pd.Series(dtype=float)).map(
        lambda v: f"{float(v):+.2f} $" if pd.notna(v) else "—"
    )

    fig = px.treemap(
        plot_df,
        path=["sector", "symbol"],
        values="size_value",
        color="metric_value",
        color_continuous_scale=color_scheme,
        range_color=(cmin, cmax),
        color_continuous_midpoint=midpoint,
        custom_data=[
            "name",
            "last_close",
            "metric_value",
            "market_cap_b",
            "color_label",
            "change_abs_label",
            "last_date",
        ],
        hover_name="symbol",
    )

    if show_symbols:
        fig.update_traces(
            text=plot_df["tile_text"],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#111827"},
            insidetextfont={"size": 11, "color": "#111827"},
        )
    else:
        fig.update_traces(texttemplate="%{label}", textinfo="label")

    fig.update_traces(
        textposition="middle center",
        marker={"line": {"width": 1.5, "color": "#ffffff"}},
        hovertemplate=(
            "<b>%{label}</b> (%{customdata[0]})<br>"
            "Secteur : %{parent}<br>"
            "Mesure : <b>%{customdata[4]}</b><br>"
            "Dernier cours : %{customdata[1]:,.2f} $<br>"
            "Variation abs. : %{customdata[5]}<br>"
            "Cap. boursière : %{customdata[3]:,.0f} Md $<br>"
            "Date : %{customdata[6]|%d/%m/%Y}<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=36, b=4),
        height=680,
        coloraxis_colorbar=dict(
            title=MEASUREMENT_LABELS.get(measurement, "Valeur"),
            ticksuffix="%" if measurement == "performance" else "",
            thickness=14,
            len=0.55,
            y=0.02,
            yanchor="bottom",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#4338ca",
            font={"size": 13, "color": "#111827"},
        ),
        uniformtext=dict(minsize=9, mode="hide"),
    )
    return fig


def _build_mini_chart(
    panel: pd.DataFrame,
    symbol: str,
    *,
    months: int = 12,
) -> go.Figure | None:
    sym_df = panel[panel["symbol"] == symbol.upper()].sort_values("date")
    if sym_df.empty:
        return None

    cutoff = sym_df["date"].max() - pd.DateOffset(months=months)
    sym_df = sym_df[sym_df["date"] >= cutoff]
    if sym_df.empty:
        return None

    start = float(sym_df.iloc[0]["close"])
    end = float(sym_df.iloc[-1]["close"])
    line_color = "#15803d" if end >= start else "#dc2626"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sym_df["date"],
            y=sym_df["close"],
            mode="lines",
            line={"color": line_color, "width": 2},
            fill="tozeroy",
            fillcolor=f"rgba({'21,128,61' if end >= start else '220,38,38'},0.08)",
            hovertemplate=(
                "%{x|%d/%m/%Y}<br>Clôture ajustée : %{y:,.2f} $<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=180,
        margin=dict(l=36, r=8, t=8, b=28),
        showlegend=False,
        xaxis=dict(showgrid=False, tickformat="%b %Y", tickfont={"size": 9}),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickformat=",.0f", tickfont={"size": 9}),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _symbol_from_selection(
    selection,
    df: pd.DataFrame,
    fig: go.Figure | None = None,
) -> str | None:
    if selection is None:
        return None
    points = getattr(selection, "points", None) or []
    if not points:
        return None

    known = set(df["symbol"].astype(str))
    trace_labels: list[str] = []
    if fig is not None and fig.data:
        raw_labels = fig.data[0].labels
        if raw_labels is not None:
            trace_labels = [str(x) for x in raw_labels]

    for pt in points:
        if isinstance(pt, dict):
            label = pt.get("label") or pt.get("x")
            custom = pt.get("customdata") or pt.get("custom_data")
            point_index = pt.get("point_index", pt.get("point_number"))
        else:
            label = getattr(pt, "label", None) or getattr(pt, "x", None)
            custom = getattr(pt, "customdata", None)
            point_index = getattr(pt, "point_index", getattr(pt, "point_number", None))

        if label and str(label) in known:
            return str(label)
        if custom is not None:
            if isinstance(custom, (list, tuple)) and len(custom) > 0:
                candidate = str(custom[0])
                if candidate in known:
                    return candidate
            elif hasattr(custom, "__len__") and len(custom) > 0:
                candidate = str(custom[0])
                if candidate in known:
                    return candidate
        if point_index is not None and trace_labels:
            idx = int(point_index)
            if 0 <= idx < len(trace_labels):
                lbl = str(trace_labels[idx])
                if lbl in known:
                    return lbl
    return None


def _render_info_panel(
    symbol: str,
    df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    measurement: str,
    show_info_box: bool,
    show_mini_chart: bool,
) -> None:
    row = df[df["symbol"] == symbol]
    if row.empty:
        st.info(f"Aucune donnée pour **{symbol}**.")
        return

    r = row.iloc[0]
    if show_info_box:
        metric_val = r["metric_value"]
        metric_txt = _format_metric(metric_val, measurement) if pd.notna(metric_val) else "—"
        metric_color = (
            _metric_color(float(metric_val), measurement)
            if pd.notna(metric_val)
            else "#6b7280"
        )

        st.markdown(
            f"""
            <div style="
                background:#fff;border:1px solid #e5e7eb;border-radius:12px;
                padding:1rem 1.1rem;box-shadow:0 8px 24px rgba(67,56,202,.12);
            ">
                <div style="font-weight:700;font-size:1rem;color:#111827;">
                    {r['name']} ({r['symbol']})
                </div>
                <div style="font-size:1.45rem;font-weight:800;color:{metric_color};margin:.35rem 0;">
                    {metric_txt}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_mini_chart:
        mini = _build_mini_chart(panel, symbol)
        if mini is not None:
            st.plotly_chart(mini, use_container_width=True, key=f"mc_mini_{symbol}")

    if show_info_box:
        last_close = r.get("last_close")
        change_abs = r.get("change_abs")
        last_date = r.get("last_date")
        st.markdown("**Détails**")
        if pd.notna(last_close):
            st.metric("Dernier cours", f"{float(last_close):,.2f} $")
        if pd.notna(change_abs):
            st.metric("Variation absolue (1 séance)", f"{float(change_abs):+.2f} $")
        if pd.notna(last_date):
            st.caption(f"Cotation au {pd.Timestamp(last_date).strftime('%d/%m/%Y')}")
        st.caption(f"Secteur : **{r['sector']}** · Cap. : **{float(r['market_cap_b']):,.0f} Md $**")


def _ohlcv_row_count() -> int | None:
    if not pg_enabled():
        return None
    try:
        row = read_sql("SELECT COUNT(*) AS n FROM stocks.ohlcv").iloc[0]["n"]
        return int(row)
    except Exception:
        return None


def _pg_connection_hint() -> str:
    from src.db.postgres import pg_connection_hint

    return pg_connection_hint()


def _render_ohlcv_error(meta: dict) -> None:
    ohlcv_n = _ohlcv_row_count()
    load_error = meta.get("load_error")

    st.error("La Market Carpet n'a pas pu charger les cours OHLCV.")

    if ohlcv_n and ohlcv_n > 0:
        st.warning(
            f"Les données **existent** dans PostgreSQL (`stocks.ohlcv` : "
            f"**{ohlcv_n:,}** lignes). Le problème vient du **chargement**, "
            "pas de l'import Dolt."
        )
    elif ohlcv_n == 0:
        st.info(
            "La table `stocks.ohlcv` est **vide**. Lancez l'import :\n\n"
            "`python scripts/dolt_to_postgres.py --repos stocks`"
        )
    else:
        st.info(
            "Impossible de joindre PostgreSQL. " + _pg_connection_hint()
        )

    if load_error:
        with st.expander("Détail technique"):
            st.code(str(load_error))

    st.caption(
        "Si vous venez de corriger la configuration, cliquez **Actualiser les données** "
        "dans la barre latérale (le cache Streamlit peut conserver un ancien échec)."
    )


def render_market_carpet() -> None:
    """Affiche la Market Carpet interactive (style StockCharts)."""
    st.markdown(
        """
        <div class="card">
            <div class="card-header">
                <strong>Market Carpet</strong>
                <span class="badge badge-insight">Carte thermique sectorielle</span>
            </div>
            <p class="card-desc">
                Visualisez d'un coup d'œil la performance des actions par secteur —
                comme le <a href="https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/marketcarpets"
                target="_blank">Market Carpet StockCharts</a>.
                La <strong>taille</strong> encode la capitalisation (ou le prix / poids égal) ;
                la <strong>couleur</strong> reflète la mesure choisie (performance, RSI, jours haussiers).
                Cliquez une case pour afficher le détail ; double-cliquez pour zoomer sur un secteur.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pg_enabled():
        st.warning(
            "PostgreSQL n'est pas configuré. La Market Carpet nécessite les cours "
            "dans `stocks.ohlcv` (import Dolt → Postgres)."
        )
        return

    groups = list_universe_groups()
    group_labels = {k: label for k, label in groups}

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        group_key = st.selectbox(
            "Univers",
            options=[k for k, _ in groups],
            format_func=lambda k: group_labels.get(k, k),
            key="mc_group",
        )
    with ctrl2:
        measurement = st.selectbox(
            "Mesure",
            options=list(MEASUREMENT_LABELS.keys()),
            format_func=lambda k: MEASUREMENT_LABELS[k],
            key="mc_measurement",
        )
    with ctrl3:
        period_key = st.selectbox(
            "Couleur (période)",
            options=list(PERIOD_LABELS.keys()),
            format_func=lambda k: PERIOD_LABELS[k],
            key="mc_period",
            help="Fenêtre utilisée pour colorer les cases (ex. 1 jour = dernière séance).",
        )
    with ctrl4:
        size_mode = st.selectbox(
            "Taille des cases",
            options=list(SIZE_LABELS.keys()),
            format_func=lambda k: SIZE_LABELS[k],
            key="mc_size",
        )

    opt1, opt2, opt3, opt4, opt5 = st.columns([1, 1, 1, 1, 1])
    with opt1:
        color_scheme = st.selectbox(
            "Palette",
            options=list(COLOR_SCHEMES.keys()),
            key="mc_palette",
        )
    with opt2:
        show_symbols = st.checkbox("Afficher symboles", value=True, key="mc_symbols")
    with opt3:
        show_info_box = st.checkbox("Encadré info (survol)", value=True, key="mc_info")
    with opt4:
        show_mini_chart = st.checkbox("Mini-graphique au clic", value=True, key="mc_mini")
    with opt5:
        auto_refresh = st.checkbox("Rafraîchissement auto (5 min)", value=False, key="mc_refresh")

    opt6, opt7 = st.columns([1, 3])
    with opt6:
        show_table = st.checkbox("Vue tableau", value=False, key="mc_table")
    with opt7:
        manual_symbol = st.selectbox(
            "Symbole (détail manuel)",
            options=["—"] + load_universe(group_key)["symbol"].tolist(),
            key="mc_manual_symbol",
            help="Alternative au clic sur la treemap pour afficher le panneau de détail.",
        )

    if auto_refresh:
        st_autorefresh(interval=300_000, key="mc_autorefresh")

    with st.spinner("Chargement des cours et calcul des indicateurs…"):
        df, meta = _cached_carpet(group_key, measurement, period_key, size_mode)
        if not meta.get("pg_available"):
            _cached_carpet.clear()
            df, meta = build_market_carpet_df(
                group_key,
                measurement=measurement,
                period_key=period_key,
                size_mode=size_mode,
            )
        panel = _cached_ohlcv_panel(group_key)
        if panel.empty:
            _cached_ohlcv_panel.clear()
            panel = load_ohlcv_panel(load_universe(group_key)["symbol"].tolist())

    if df.empty or not meta.get("pg_available"):
        _render_ohlcv_error(meta)
        return

    plot_df = df.dropna(subset=["metric_value", "size_value"])
    if plot_df.empty:
        st.info("Données insuffisantes pour calculer les métriques sur cet univers.")
        return

    n_ok = meta.get("symbols_with_data", len(plot_df))
    n_total = meta.get("symbols_total", len(plot_df))
    data_to = meta.get("data_to")
    date_str = (
        pd.Timestamp(data_to).strftime("%d/%m/%Y")
        if data_to is not None and pd.notna(data_to)
        else "—"
    )
    period_label = PERIOD_LABELS.get(period_key, period_key)
    measure_label = MEASUREMENT_LABELS.get(measurement, measurement)

    st.caption(
        f"{n_ok}/{n_total} symboles avec données · "
        f"Dernière cotation : {date_str} · "
        f"{measure_label} · {period_label} · "
        f"Taille : {SIZE_LABELS.get(size_mode, size_mode)}"
    )

    selected_symbol: str | None = None
    if manual_symbol and manual_symbol != "—":
        selected_symbol = manual_symbol

    if show_table:
        table = plot_df[
            ["symbol", "name", "sector", "metric_value", "last_close", "market_cap_b"]
        ].copy()
        table = table.rename(columns={
            "symbol": "Symbole",
            "name": "Nom",
            "sector": "Secteur",
            "metric_value": "Mesure",
            "last_close": "Cours ($)",
            "market_cap_b": "Cap. (Md $)",
        })
        table["Mesure"] = table["Mesure"].map(
            lambda v: _format_metric(v, measurement) if pd.notna(v) else "—"
        )
        table["Cours ($)"] = table["Cours ($)"].map(
            lambda v: f"{v:,.2f}" if pd.notna(v) else "—"
        )
        st.dataframe(
            table.sort_values(["Secteur", "Symbole"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        show_side = show_info_box or show_mini_chart
        if show_side:
            treemap_col, info_col = st.columns([2.8, 1])
        else:
            treemap_col = st.container()
            info_col = None

        with treemap_col:
            fig = _build_treemap(
                plot_df,
                measurement=measurement,
                period_key=period_key,
                color_scheme=COLOR_SCHEMES[color_scheme],
                show_symbols=show_symbols,
            )
            chart_event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="mc_treemap",
            )
            clicked = _symbol_from_selection(
                getattr(chart_event, "selection", None),
                plot_df,
                fig,
            )
            if clicked:
                selected_symbol = clicked

        if info_col is not None:
            with info_col:
                st.markdown("##### Détail du titre")
                if selected_symbol:
                    _render_info_panel(
                        selected_symbol,
                        plot_df,
                        panel,
                        measurement=measurement,
                        show_info_box=show_info_box,
                        show_mini_chart=show_mini_chart,
                    )
                else:
                    st.caption(
                        "Survolez une case pour l'infobulle Plotly, "
                        "ou **cliquez** une case / choisissez un symbole ci-dessus "
                        "pour afficher le mini-graphique et les détails."
                    )

    sector_perf = (
        plot_df.groupby("sector", as_index=False)["metric_value"]
        .mean()
        .sort_values("metric_value", ascending=False)
    )
    if not sector_perf.empty:
        st.markdown("**Performance moyenne par secteur**")
        cols = st.columns(min(len(sector_perf), 4))
        for idx, row in enumerate(sector_perf.itertuples()):
            with cols[idx % len(cols)]:
                st.metric(
                    row.sector,
                    _format_metric(row.metric_value, measurement),
                )

    st.caption(
        "Inspiré du Market Carpet StockCharts — "
        "double-cliquez une case pour zoomer sur un secteur. "
        f"Période technique : {PERIOD_DAYS.get(period_key, 1)} séances."
    )
