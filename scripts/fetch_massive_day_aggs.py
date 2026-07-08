"""Telecharge les Day Aggregates Massive (flat files S3)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv  # noqa: E402

load_dotenv()

from src.bronze.massive import (  # noqa: E402
    download_range,
    get_s3_client,
    list_available,
)
from src.config import MASSIVE_CACHE_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Telecharge les Day Aggregates Massive")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="Lister les fichiers disponibles")
    list_parser.add_argument("year", type=int)
    list_parser.add_argument("month", type=int, nargs="?", default=None)

    dl_parser = sub.add_parser("download", help="Telecharger un ou plusieurs jours")
    dl_parser.add_argument("--date", type=str, help="Date unique YYYY-MM-DD")
    dl_parser.add_argument("--from", dest="date_from", type=str, help="Debut YYYY-MM-DD")
    dl_parser.add_argument("--to", dest="date_to", type=str, help="Fin YYYY-MM-DD")
    dl_parser.add_argument(
        "--output-dir",
        type=Path,
        default=MASSIVE_CACHE_DIR,
        help="Dossier de sortie",
    )

    args = parser.parse_args()

    try:
        client = get_s3_client()
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "list":
        keys = list_available(client, args.year, args.month)
        if not keys:
            print("Aucun fichier trouve.")
            return
        for key in keys:
            print(key)
        print(f"\n{len(keys)} fichier(s)")
        return

    if args.date:
        start = end = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.date_from and args.date_to:
        start = datetime.strptime(args.date_from, "%Y-%m-%d").date()
        end = datetime.strptime(args.date_to, "%Y-%m-%d").date()
    else:
        dl_parser.error("Precisez --date ou --from / --to")

    print(f"Telechargement {start} -> {end} vers {args.output_dir}")
    paths = download_range(start, end, args.output_dir, client)
    print(f"\nTermine : {len(paths)} fichier(s) telecharge(s)")


if __name__ == "__main__":
    main()
