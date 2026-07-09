#!/usr/bin/env python3
"""Importe le depot Dolt rates vers PostgreSQL (schema: rates)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.postgres import engine, pg_enabled, test_connection  # noqa: E402
from src.dolt.postgres_import import import_repo_to_postgres  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importe le depot Dolt rates vers PostgreSQL (schema rates).",
    )
    parser.add_argument(
        "--dolt-root",
        type=Path,
        default=Path("/data"),
        help="Dossier contenant le clone rates/ (defaut: /data dans ubuntu-box)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        help="Limiter a certaines tables du depot rates.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Lignes par lot a l'import PostgreSQL (defaut: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lister les tables sans importer",
    )
    args = parser.parse_args()

    if not args.dry_run and not pg_enabled():
        print(
            "Erreur : PGHOST et PGPASSWORD requis (ex. docker compose + variables PG*).",
            file=sys.stderr,
        )
        return 1

    dolt_root = args.dolt_root.expanduser().resolve()
    if not dolt_root.is_dir():
        print(f"Erreur : dossier Dolt introuvable : {dolt_root}", file=sys.stderr)
        return 1

    repo_dir = dolt_root / "rates"
    schema = "rates"

    if args.dry_run:
        tables = import_repo_to_postgres(
            repo_dir=repo_dir,
            schema=schema,
            eng=None,
            chunksize=args.chunksize,
            tables=args.tables,
            dry_run=True,
        )
        print(f"[dry-run] rates -> schema {schema} : {len(tables)} table(s)")
        for t in tables.keys():
            print(f"  - {t}")
        return 0

    eng = engine()
    test_connection(eng=eng)
    print(f"PostgreSQL OK — import rates depuis {dolt_root}\n")

    imported = import_repo_to_postgres(
        repo_dir=repo_dir,
        schema=schema,
        eng=eng,
        chunksize=args.chunksize,
        tables=args.tables,
        dry_run=False,
    )

    total = sum(imported.values())
    print("\nTermine.")
    print(f"  {schema}: {len(imported)} table(s), {total:,} ligne(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

