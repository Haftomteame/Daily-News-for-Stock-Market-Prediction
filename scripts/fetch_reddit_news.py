"""Telecharge des headlines Reddit vers lakehouse/bronze/news_reddit/."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bronze.reddit_fetch import (  # noqa: E402
    RedditFetchError,
    fetch_range_arctic,
    fetch_recent_praw,
)
from src.bronze.storage import bronze_exists, read_bronze_data, write_bronze  # noqa: E402
from src.env import load_dotenv  # noqa: E402
from src.storage.io import join_path, makedirs  # noqa: E402


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Fetch Reddit headlines vers lakehouse/bronze/news_reddit/"
    )
    parser.add_argument("--from", dest="date_from", help="YYYY-MM-DD (requis sauf --praw)")
    parser.add_argument("--to", dest="date_to", help="YYYY-MM-DD (requis sauf --praw)")
    parser.add_argument(
        "--subreddit",
        default="stocks,wallstreetbets,StockMarket,investing",
        help="Subreddit(s) separes par virgule (defaut: stocks,wallstreetbets,StockMarket,investing)",
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=None,
        help="Limite de posts par jour et par subreddit (test rapide)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Limite totale de posts",
    )
    parser.add_argument(
        "--praw",
        action="store_true",
        help="Utiliser PRAW (posts recents seulement, pas historique)",
    )
    parser.add_argument(
        "--table",
        default="news_reddit",
        help="Table Bronze cible (defaut: news_reddit). Un nom par subreddit en parallele.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Ajouter aux donnees Bronze existantes (dedupe Date+News)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ne pas re-fetcher les dates deja presentes dans la table Bronze (--append requis)",
    )
    parser.add_argument(
        "--partial",
        type=Path,
        default=None,
        help="Sauvegarde partielle CSV en cas d'erreur (debug)",
    )
    args = parser.parse_args()

    subreddits = [s.strip() for s in args.subreddit.split(",") if s.strip()]
    partial = (
        str(args.partial)
        if args.partial
        else join_path("lakehouse", "bronze", args.table, "partial.csv")
    )
    makedirs(partial)

    try:
        if args.praw:
            frames = [fetch_recent_praw(sub) for sub in subreddits]
            import pandas as pd

            df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "News"])
            source_label = "praw"
        else:
            if not args.date_from or not args.date_to:
                parser.error("--from et --to requis (ou utilisez --praw)")

            start = datetime.strptime(args.date_from, "%Y-%m-%d").date()
            end = datetime.strptime(args.date_to, "%Y-%m-%d").date()

            if args.skip_existing:
                if not args.append:
                    parser.error("--skip-existing requiert --append")
                if bronze_exists(args.table):
                    import pandas as pd

                    existing = read_bronze_data(args.table)
                    max_day = pd.to_datetime(existing["Date"], errors="coerce").max().date()
                    if max_day >= start:
                        new_start = max_day + timedelta(days=1)
                        if new_start > end:
                            print(
                                f"Skip : {args.table} couvre deja jusqu'a {max_day} "
                                f"(demande {start} -> {end}). Rien a fetcher."
                            )
                            sys.exit(0)
                        print(
                            f"Skip existant : {start} -> {max_day} deja en bronze, "
                            f"fetch a partir de {new_start}"
                        )
                        start = new_start

            def progress(day, day_count, total):
                print(f"  {day}: +{day_count} (total {total})")

            print(f"Fetch Reddit {start} -> {end} | subs: {', '.join(subreddits)} | table: {args.table}")
            df = fetch_range_arctic(
                start,
                end,
                subreddits=subreddits,
                max_per_day=args.max_per_day,
                max_total=args.max_total,
                on_progress=progress,
                partial_output=partial,
            )
            source_label = "arctic_shift"
    except RedditFetchError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd

    if args.append and bronze_exists(args.table):
        existing = read_bronze_data(args.table)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["Date", "News"])
            .sort_values(["Date", "News"])
            .reset_index(drop=True)
        )
        print(f"Append : {len(df):,} lignes apres fusion avec l'existant")

    _, path = write_bronze(df, args.table, str(uuid.uuid4()), source_label)
    print(f"\nOK {len(df):,} lignes -> {path}")
    print(f"Periode : {df['Date'].min()} -> {df['Date'].max()}")


if __name__ == "__main__":
    main()
