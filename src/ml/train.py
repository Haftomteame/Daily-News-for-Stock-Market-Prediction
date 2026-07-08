"""Entrainement ML — prediction Label (direction DJIA) depuis couche Gold."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

FEATURE_COLUMNS = [
    "daily_return_pct",
    "volatility_5d",
    "news_count",
    "avg_headline_length",
    "finance_news_ratio",
    "combined_finance_ratio",
    "combined_avg_length",
]


def _load_training_data() -> pd.DataFrame:
    gold_path = gold_data_path()
    combined_path = silver_data_path("news_combined")

    if not exists(gold_path) or not exists(combined_path):
        raise FileNotFoundError("Couches Gold et Silver (news_combined) requises pour le ML.")

    df = query_duckdb(
        """
        SELECT
            g.date,
            g.daily_return_pct,
            g.volatility_5d,
            g.news_count,
            g.avg_headline_length,
            g.finance_news_ratio,
            c._combined_finance_ratio AS combined_finance_ratio,
            c._combined_avg_length AS combined_avg_length,
            c.Label AS label
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
    return df


def _temporal_split(df: pd.DataFrame, prediction_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"].dt.year < prediction_year].copy()
    predict_df = df[df["date"].dt.year == prediction_year].copy()
    return train_df, predict_df


def run_ml_training(
    batch_id: str,
    prediction_year: int | None = None,
    random_state: int = 42,
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

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["label"].astype(int)
    X_predict = predict_df[FEATURE_COLUMNS].fillna(0)
    y_true = predict_df["label"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_predict)

    train_start = train_df["date"].min().date().isoformat()
    train_end = train_df["date"].max().date().isoformat()
    predict_start = predict_df["date"].min().date().isoformat()
    predict_end = predict_df["date"].max().date().isoformat()

    metrics = {
        "batch_id": batch_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "prediction_year": prediction_year,
        "train_period": f"{train_start} -> {train_end}",
        "predict_period": f"{predict_start} -> {predict_end}",
        "samples_total": len(df),
        "samples_train": len(train_df),
        "samples_predict": len(predict_df),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "features": FEATURE_COLUMNS,
        "target": "label",
        "model": "LogisticRegression",
        "split": "temporal",
    }

    write_joblib(model, ML_MODEL_PATH)
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
