"""Tests module ML."""

import json

from src.bronze.ingest import run_bronze_ingestion
from src.config import ML_METRICS_PATH, ML_MODEL_PATH
from src.gold.aggregate import run_gold_aggregate
from src.ml.train import run_ml_training
from src.silver.transform import run_silver_transform


def test_ml_training(project_root):
    run_bronze_ingestion("test-batch")
    run_silver_transform("test-batch")
    run_gold_aggregate("test-batch")

    # Dataset mini trop petit pour ML — on verifie l'erreur attendue
    try:
        run_ml_training("test-batch")
    except ValueError as exc:
        assert "trop petit" in str(exc).lower()
        return

    # Si le split passe (peu probable avec 3 lignes), verifier les artefacts
    assert ML_MODEL_PATH.exists()
    metrics = json.loads(ML_METRICS_PATH.read_text(encoding="utf-8"))
    assert "accuracy" in metrics
