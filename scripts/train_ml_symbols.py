#!/usr/bin/env python3
"""Entraine un modele ML par symbole (PostgreSQL OHLCV + Reddit)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAINING_SYMBOLS  # noqa: E402
from src.env import load_dotenv  # noqa: E402
from src.ml.train import run_ml_training, run_ml_training_all  # noqa: E402

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entraine les modeles de direction (hausse/baisse) par symbole.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help=f"Symboles a entrainer (defaut: {', '.join(TRAINING_SYMBOLS)}).",
    )
    parser.add_argument(
        "--symbol",
        metavar="SYM",
        help="Entrainer un seul symbole.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Annee de holdout pour l'evaluation (defaut: ML_PREDICTION_YEAR).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Desactiver le tuning des hyperparametres.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Identifiant de lot (defaut: UUID).",
    )
    args = parser.parse_args()

    batch_id = args.batch_id or str(uuid.uuid4())

    if args.symbol:
        result = run_ml_training(
            batch_id,
            prediction_year=args.year,
            symbol=args.symbol.upper(),
            tune=not args.no_tune,
        )
        print(f"OK {args.symbol.upper()} — accuracy {result['metrics']['accuracy']:.0%}")
        print(f"  Modele : {result['model_path']}")
        return 0

    symbols = [s.upper() for s in (args.symbols or TRAINING_SYMBOLS)]
    summary = run_ml_training_all(
        batch_id,
        symbols=symbols,
        prediction_year=args.year,
        tune=not args.no_tune,
    )
    print(f"\nTermine — {len(summary['symbols_ok'])}/{len(symbols)} symboles entraines.")
    if summary["symbols_failed"]:
        print("Echecs :")
        for sym, err in summary["errors"].items():
            print(f"  {sym}: {err}")
        return 1 if not summary["symbols_ok"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
