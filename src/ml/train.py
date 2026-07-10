"""Entrainement ML — prediction direction (hausse/baisse) par symbole."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.bronze.build_combined import aggregate_reddit_tops, compute_labels
from src.config import (
    MASSIVE_TICKER,
    ML_PREDICTION_YEAR,
    TRAINING_SYMBOLS,
    bronze_data_path,
    gold_data_path,
    silver_data_path,
)
from src.db.postgres import pg_enabled, read_sql
from src.silver.transform import _has_finance_keyword
from src.storage.io import exists, query_duckdb, write_json, write_joblib, write_parquet
from src.storage.paths import ml_metrics_path, ml_model_path, ml_predictions_path

TOP_COLS = [f"Top{i}" for i in range(1, 26)]
NUMERIC_FEATURES = [
    "daily_return_pct",
    "volatility_5d",
    "news_count",
    "finance_news_ratio",
]

TFIDF_MAX_FEATURES = 300
TFIDF_MIN_DF = 3
LOGISTIC_C = 0.1

FEATURE_DESCRIPTION = [
    "headlines_tfidf (Top1..Top25)",
    *NUMERIC_FEATURES,
]

REMOVED_FEATURES = [
    "avg_headline_length",
    "combined_finance_ratio",
    "combined_avg_length",
]

TUNE_GRID = {
    "max_features": [200, 300, 500, 800],
    "min_df": [2, 3, 5],
    "C": [0.01, 0.05, 0.1, 0.3, 0.5],
}


def _concat_headlines(row: pd.Series) -> str:
    parts: list[str] = []
    for col in TOP_COLS:
        if col not in row.index:
            continue
        text = str(row[col]).strip()
        if text and text.lower() != "nan":
            parts.append(text)
    return " ".join(parts)


def _load_training_data_lakehouse() -> pd.DataFrame:
    """Dataset legacy DIA depuis Gold + Silver combined."""
    gold_path = gold_data_path()
    combined_path = silver_data_path("news_combined")

    if not exists(gold_path) or not exists(combined_path):
        raise FileNotFoundError("Couches Gold et Silver (news_combined) requises pour le ML.")

    top_select = ", ".join(f'c."{col}"' for col in TOP_COLS)
    df = query_duckdb(
        f"""
        SELECT
            g.date,
            g.daily_return_pct,
            g.volatility_5d,
            g.news_count,
            g.finance_news_ratio,
            c.Label AS label,
            {top_select}
        FROM gold_kpis g
        INNER JOIN silver_combined c
            ON CAST(c.Date AS DATE) = g.date
        WHERE c.Label IN (0, 1)
          AND c._quality_score = 1
        ORDER BY g.date
        """,
        {"gold_kpis": gold_path, "silver_combined": combined_path},
    )
    df["date"] = pd.to_datetime(df["date"])
    df["headline_text"] = df.apply(_concat_headlines, axis=1)
    return df[df["headline_text"].str.len() > 0].copy()


def _reddit_data_path() -> str | None:
    silver = silver_data_path("news_reddit")
    if exists(silver):
        return silver
    bronze = bronze_data_path("news_reddit")
    if exists(bronze):
        return bronze
    return None


def _load_reddit_for_symbol(symbol: str) -> pd.DataFrame:
    """Charge Reddit pour l'entrainement (titres generaux — couverture temporelle complete)."""
    path = _reddit_data_path()
    if not path:
        raise FileNotFoundError("Couche news_reddit absente (bronze ou silver).")

    return query_duckdb(
        """
        SELECT CAST("Date" AS DATE) AS Date, CAST(News AS VARCHAR) AS News
        FROM data_table
        WHERE News IS NOT NULL AND TRIM(CAST(News AS VARCHAR)) != ''
        """,
        {"data_table": path},
    )


def _daily_news_kpis(reddit: pd.DataFrame) -> pd.DataFrame:
    df = reddit.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["_finance"] = df["News"].astype(str).apply(_has_finance_keyword)
    agg = (
        df.groupby("Date", as_index=False)
        .agg(
            news_count=("News", "count"),
            finance_news_ratio=("_finance", "mean"),
        )
        .rename(columns={"Date": "date"})
    )
    return agg


def _load_ohlcv_kpis(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    if not pg_enabled():
        return pd.DataFrame()

    raw = read_sql(
        """
        SELECT date, open, high, low, close, volume
        FROM stocks.ohlcv
        WHERE act_symbol = :symbol
        ORDER BY date
        """,
        params={"symbol": symbol},
    )
    if raw.empty:
        return pd.DataFrame()

    stock = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")
    stock = stock.dropna(subset=["Date"]).sort_values("Date")

    labeled = compute_labels(stock)
    labeled["date"] = pd.to_datetime(labeled["Date"])
    labeled["daily_return_pct"] = labeled["Close"].pct_change() * 100
    labeled["volatility_5d"] = labeled["daily_return_pct"].rolling(5, min_periods=1).std()
    return labeled


def _load_training_data(symbol: str | None = None) -> pd.DataFrame:
    """Charge le dataset d'entrainement pour un symbole (PostgreSQL + Reddit)."""
    symbol = (symbol or MASSIVE_TICKER).upper()

    if symbol == MASSIVE_TICKER and not pg_enabled():
        return _load_training_data_lakehouse()

    ohlcv = _load_ohlcv_kpis(symbol)
    if ohlcv.empty:
        if symbol == MASSIVE_TICKER:
            return _load_training_data_lakehouse()
        raise FileNotFoundError(f"Aucun OHLCV PostgreSQL pour {symbol}.")

    reddit = _load_reddit_for_symbol(symbol)
    tops = aggregate_reddit_tops(reddit)
    tops["date"] = pd.to_datetime(tops["Date"])
    news_kpis = _daily_news_kpis(reddit)

    df = ohlcv.merge(tops, on="date", how="inner", suffixes=("", "_top"))
    df = df.merge(news_kpis, on="date", how="left")
    df["news_count"] = df["news_count"].fillna(0)
    df["finance_news_ratio"] = df["finance_news_ratio"].fillna(0)
    if "label" not in df.columns and "Label" in df.columns:
        df["label"] = df["Label"]
    df["headline_text"] = df.apply(_concat_headlines, axis=1)
    df = df[df["headline_text"].str.len() > 0].copy()
    df = df[df["label"].isin([0, 1])].sort_values("date")
    return df


def _temporal_split(
    df: pd.DataFrame,
    prediction_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    available_years = sorted(int(y) for y in df["date"].dt.year.unique())
    year = prediction_year
    if year not in available_years:
        year = available_years[-1]

    train_df = df[df["date"].dt.year < year].copy()
    predict_df = df[df["date"].dt.year == year].copy()

    if predict_df.empty and len(available_years) >= 2:
        year = available_years[-1]
        train_df = df[df["date"].dt.year < year].copy()
        predict_df = df[df["date"].dt.year == year].copy()

    return train_df, predict_df, year


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["headline_text", *NUMERIC_FEATURES]
    out = df[feature_cols].copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(0)
    return out


def _build_model(
    random_state: int,
    *,
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int = TFIDF_MIN_DF,
    C: float = LOGISTIC_C,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
                "headline_text",
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocessor),
        (
            "clf",
            LogisticRegression(
                max_iter=3000,
                random_state=random_state,
                class_weight="balanced",
                C=C,
                solver="lbfgs",
            ),
        ),
    ])


def _classification_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _tune_hyperparameters(
    train_df: pd.DataFrame,
    random_state: int,
    n_splits: int = 5,
) -> dict[str, float | int]:
    X = _feature_matrix(train_df)
    y = train_df["label"].astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score = -1.0
    best_params = {
        "max_features": TFIDF_MAX_FEATURES,
        "min_df": TFIDF_MIN_DF,
        "C": LOGISTIC_C,
    }

    for max_features, min_df, C in product(
        TUNE_GRID["max_features"],
        TUNE_GRID["min_df"],
        TUNE_GRID["C"],
    ):
        fold_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            model = _build_model(
                random_state,
                max_features=max_features,
                min_df=min_df,
                C=C,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            fold_scores.append(f1_score(y.iloc[val_idx], pred, zero_division=0))

        mean_f1 = sum(fold_scores) / len(fold_scores)
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = {
                "max_features": max_features,
                "min_df": min_df,
                "C": C,
                "cv_f1": round(mean_f1, 4),
            }

    return best_params


def _write_legacy_dia_artifacts(
    model: Pipeline,
    metrics: dict,
    predictions: pd.DataFrame,
) -> None:
    """Compatibilite : copie les artefacts DIA vers lakehouse/ml/ (chemins historiques)."""
    from src.config import ML_METRICS_PATH, ML_MODEL_PATH, ML_PREDICTIONS_PATH

    write_joblib(model, ML_MODEL_PATH)
    write_json(metrics, ML_METRICS_PATH)
    write_parquet(predictions, ML_PREDICTIONS_PATH)


def run_ml_training(
    batch_id: str,
    prediction_year: int | None = None,
    random_state: int = 42,
    *,
    symbol: str | None = None,
    tune: bool = True,
    retrain_on_all: bool = True,
) -> dict:
    symbol = (symbol or MASSIVE_TICKER).upper()
    prediction_year = prediction_year or ML_PREDICTION_YEAR
    df = _load_training_data(symbol)
    train_df, predict_df, prediction_year = _temporal_split(df, prediction_year)

    if len(train_df) < 50:
        raise ValueError(
            f"Dataset d'entrainement trop petit pour {symbol} ({len(train_df)} lignes) "
            f"pour prediction_year={prediction_year}."
        )
    if predict_df.empty:
        available = sorted(int(y) for y in df["date"].dt.year.unique())
        raise ValueError(
            f"Aucune donnee pour predire {symbol} en {prediction_year}. "
            f"Annees disponibles : {available}. "
            f"Le dataset s'arrete en {df['date'].max().date()}."
        )

    best_params: dict | None = None
    if tune and len(train_df) >= 80:
        best_params = _tune_hyperparameters(train_df, random_state)
        model = _build_model(
            random_state,
            max_features=int(best_params["max_features"]),
            min_df=int(best_params["min_df"]),
            C=float(best_params["C"]),
        )
    else:
        model = _build_model(random_state)

    X_train = _feature_matrix(train_df)
    y_train = train_df["label"].astype(int)
    X_predict = _feature_matrix(predict_df)
    y_true = predict_df["label"].astype(int)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_predict)

    train_start = train_df["date"].min().date().isoformat()
    train_end = train_df["date"].max().date().isoformat()
    predict_start = predict_df["date"].min().date().isoformat()
    predict_end = predict_df["date"].max().date().isoformat()

    holdout_metrics = _classification_metrics(y_true, y_pred)
    train_metrics = _classification_metrics(y_train, y_train_pred)

    production_model = model
    production_samples = len(train_df)
    full_period = f"{train_start} -> {train_end}"

    if retrain_on_all:
        production_model = _build_model(
            random_state,
            max_features=int((best_params or {})["max_features"]) if best_params else TFIDF_MAX_FEATURES,
            min_df=int((best_params or {})["min_df"]) if best_params else TFIDF_MIN_DF,
            C=float((best_params or {})["C"]) if best_params else LOGISTIC_C,
        )
        X_all = _feature_matrix(df)
        y_all = df["label"].astype(int)
        production_model.fit(X_all, y_all)
        production_samples = len(df)
        full_period = f"{df['date'].min().date().isoformat()} -> {df['date'].max().date().isoformat()}"

    metrics = {
        "batch_id": batch_id,
        "symbol": symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "prediction_year": prediction_year,
        "train_period": f"{train_start} -> {train_end}",
        "predict_period": f"{predict_start} -> {predict_end}",
        "production_period": full_period,
        "samples_total": len(df),
        "samples_train": len(train_df),
        "samples_predict": len(predict_df),
        "production_samples": production_samples,
        **holdout_metrics,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "features": FEATURE_DESCRIPTION,
        "removed_features": REMOVED_FEATURES,
        "best_params": best_params,
        "target": "label",
        "model": "LogisticRegression + TF-IDF (tuned)",
        "split": "temporal",
        "metric_scope": "holdout_prediction_year",
        "retrain_on_all": retrain_on_all,
    }

    model_path = ml_model_path(symbol)
    metrics_path = ml_metrics_path(symbol)
    predictions_path = ml_predictions_path(symbol)

    write_joblib(production_model, model_path)
    write_json(metrics, metrics_path)

    predictions = predict_df.copy()
    predictions["symbol"] = symbol
    predictions["predicted_label"] = y_pred
    predictions["probability_up"] = model.predict_proba(X_predict)[:, 1]
    predictions["correct"] = predictions["predicted_label"] == predictions["label"]
    predictions["_batch_id"] = batch_id
    predictions["_split"] = "predict"
    write_parquet(predictions, predictions_path)

    if symbol == MASSIVE_TICKER:
        _write_legacy_dia_artifacts(production_model, metrics, predictions)

    return {
        "batch_id": batch_id,
        "symbol": symbol,
        "metrics": metrics,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "rows": len(predictions),
        "prediction_year": prediction_year,
    }


def run_ml_training_all(
    batch_id: str,
    symbols: list[str] | None = None,
    prediction_year: int | None = None,
    *,
    tune: bool = True,
    retrain_on_all: bool = True,
) -> dict:
    """Entraine un modele par symbole et retourne le resume."""
    symbols = [s.upper() for s in (symbols or TRAINING_SYMBOLS)]
    trained: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for symbol in symbols:
        print(f"[ml] Entrainement {symbol} ...", flush=True)
        try:
            trained[symbol] = run_ml_training(
                batch_id,
                prediction_year=prediction_year,
                symbol=symbol,
                tune=tune,
                retrain_on_all=retrain_on_all,
            )
            acc = trained[symbol]["metrics"]["accuracy"]
            print(f"  OK {symbol} — accuracy holdout {acc:.0%}", flush=True)
        except (ValueError, FileNotFoundError) as exc:
            errors[symbol] = str(exc)
            print(f"  SKIP {symbol} — {exc}", flush=True)

    return {
        "batch_id": batch_id,
        "trained": trained,
        "errors": errors,
        "symbols_ok": list(trained.keys()),
        "symbols_failed": list(errors.keys()),
    }
