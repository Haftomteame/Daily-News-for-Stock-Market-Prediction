"""Telecharge les barres journalieres Massive vers lakehouse/bronze/stock_prices/."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bronze.massive_rest import MassiveRestError, fetch_daily_bars  # noqa: E402
from src.bronze.storage import write_bronze  # noqa: E402
from src.config import MASSIVE_TICKER  # noqa: E402
from src.env import load_dotenv  # noqa: E402


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch OHLCV Massive REST -> lakehouse/bronze/")
    parser.add_argument("--ticker", default=MASSIVE_TICKER, help="Symbole (defaut: DIA)")
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.strptime(args.date_from, "%Y-%m-%d").date()
    end = datetime.strptime(args.date_to, "%Y-%m-%d").date()

    try:
        df = fetch_daily_bars(args.ticker, start, end)
    except MassiveRestError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    source_label = f"massive_rest_{args.ticker.lower()}"
    _, path = write_bronze(df, "stock_prices", str(uuid.uuid4()), source_label)
    print(f"OK {len(df)} lignes -> {path}")


if __name__ == "__main__":
    main()
