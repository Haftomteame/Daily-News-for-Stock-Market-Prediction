"""Stream Finnhub WebSocket (trades) -> OHLCV 1 min (lakehouse + CSV optionnel)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bronze.finnhub_stream import (  # noqa: E402
    FinnhubStreamError,
    build_streamer,
    has_finnhub_token,
)
from src.config import FINNHUB_BUCKET_SEC, FINNHUB_TICKER, LEGACY_DATA_DIR  # noqa: E402
from src.env import load_dotenv  # noqa: E402


def default_out_path(ticker: str) -> Path:
    return LEGACY_DATA_DIR / f"finnhub_{ticker.lower()}_1m.csv"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Stream Finnhub trades -> bougies OHLCV 1 min (WebSocket)"
    )
    parser.add_argument("--ticker", default=FINNHUB_TICKER, help="Symbole (defaut: DIA)")
    parser.add_argument(
        "--bucket-sec",
        type=int,
        default=FINNHUB_BUCKET_SEC,
        help="Taille de fenetre en secondes (defaut: 60 = 1 min)",
    )
    parser.add_argument(
        "--lakehouse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ecrire dans lakehouse/bronze/stock_prices_1m/ (defaut: oui)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Ecrire aussi dans Data/finnhub_<ticker>_1m.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Chemin CSV personnalise (implique --csv)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Nombre de bougies avant flush Parquet bronze (defaut: 1)",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=None,
        help="Arreter apres N bougies (tests / debug)",
    )
    args = parser.parse_args()

    if not has_finnhub_token():
        print(
            "Erreur : FINNHUB_TOKEN manquant dans .env "
            "(https://finnhub.io/ > Dashboard > API Key).",
            file=sys.stderr,
        )
        sys.exit(1)

    out_csv = None
    if args.csv or args.out is not None:
        out_csv = args.out or default_out_path(args.ticker)

    try:
        streamer = build_streamer(
            args.ticker,
            bucket_sec=args.bucket_sec,
            out_csv=out_csv,
            lakehouse=args.lakehouse,
            flush_every=args.flush_every,
            max_candles=args.max_candles,
        )
        count = streamer.run()
    except FinnhubStreamError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    targets = []
    if args.lakehouse:
        targets.append("lakehouse/bronze/stock_prices_1m/")
    if out_csv is not None:
        targets.append(str(out_csv))
    print(f"OK {count} bougie(s) -> {', '.join(targets) or 'aucune sortie'}")


if __name__ == "__main__":
    main()
