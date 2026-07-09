"""Entrainement ML — prediction Label (direction DJIA) depuis headlines + KPIs Gold."""

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

from src.config import (
    ML_METRICS_PATH,
    ML_MODEL_PATH,
    ML_PREDICTION_YEAR,
    ML_PREDICTIONS_PATH,
    gold_data_path,
    silver_data_path,
)
from src.storage.io import exists, query_duckdb, write_json, write_joblib, write_parquet

TOP_COLS = [f"Top{i}" for i in range(1, 26)]
NUMERIC_FEATURES = [
    "daily_return_pct",
    "volatility_5d",
    "news_count",
    "finance_news_ratio",
]

# Valeurs par defaut (surchargees par le tuning)
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


def _load_training_data() -> pd.DataFrame:
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
    df = df[df["headline_text"].str.len() > 0].copy()
    return df


def _temporal_split(df: pd.DataFrame, prediction_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"].dt.year < prediction_year].copy()
    predict_df = df[df["date"].dt.year == prediction_year].copy()
    return train_df, predict_df


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
    """Validation croisee temporelle sur la periode d'entrainement."""
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


def run_ml_training(
    batch_id: str,
    prediction_year: int | None = None,
    random_state: int = 42,
    *,
    tune: bool = True,
    retrain_on_all: bool = True,
) -> dict:
    prediction_year = prediction_year or ML_PREDICTION_YEAR
    df = _load_training_data()
    train_df, predict_df = _temporal_split(df, prediction_year)

    if len(train_df) < 50:
        raise ValueError(
            f"Dataset d'entrainement trop petit ({len(train_df)} lignes) "
            f"pour prediction_year={prediction_year}."
        )
    if predict_df.empty:
        available = sorted(df["date"].dt.year.unique().tolist())
        raise ValueError(
            f"Aucune donnee pour predire en {prediction_year}. "
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

    write_joblib(production_model, ML_MODEL_PATH)
    write_json(metrics, ML_METRICS_PATH)

    predictions = predict_df.copy()
    predictions["predicted_label"] = y_pred
    predictions["probability_up"] = model.predict_proba(X_predict)[:, 1]
    predictions["correct"] = predictions["predicted_label"] == predictions["label"]
    predictions["_batch_id"] = batch_id
    predictions["_split"] = "predict"
    write_parquet(predictions, ML_PREDICTIONS_PATH)

    return {
        "batch_id": batch_id,
        "metrics": metrics,
        "model_path": ML_MODEL_PATH,
        "metrics_path": ML_METRICS_PATH,
        "predictions_path": ML_PREDICTIONS_PATH,
        "rows": len(predictions),
        "prediction_year": prediction_year,
    }
