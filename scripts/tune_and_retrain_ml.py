#!/usr/bin/env python3
"""Tune hyperparametres puis re-entraine le modele ML sur toutes les donnees."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv  # noqa: E402
from src.ml.train import run_ml_training  # noqa: E402


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tune + re-entraine le modele ML")
    parser.add_argument("--predict-year", type=int, default=2026)
    parser.add_argument(
        "--no-full-retrain",
        action="store_true",
        help="Ne pas re-entrainer sur toutes les donnees apres l'evaluation holdout",
    )
    parser.add_argument("--no-tune", action="store_true", help="Desactiver le tuning")
    args = parser.parse_args()

    batch_id = str(uuid.uuid4())
    result = run_ml_training(
        batch_id,
        prediction_year=args.predict_year,
        tune=not args.no_tune,
        retrain_on_all=not args.no_full_retrain,
    )
    m = result["metrics"]
    print(f"\n=== Holdout {m['prediction_year']} ===")
    print(f"Accuracy  : {m['accuracy']:.1%}")
    print(f"Precision : {m['precision']:.1%}")
    print(f"Recall    : {m['recall']:.1%}")
    print(f"F1        : {m['f1']:.1%}")
    if m.get("best_params"):
        print(f"Best params: {m['best_params']}")
    if m.get("production_samples"):
        print(f"\nModele production entraine sur {m['production_samples']} jours (toutes donnees)")
    print(f"\nArtefacts -> {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
