#!/usr/bin/env python3
"""Importe des depots Dolt clones vers PostgreSQL (un schema par depot)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.db.postgres import (  # noqa: E402
    ensure_schema,
    ensure_warehouse_indexes,
    engine,
    pg_enabled,
    test_connection,
)
from src.dolt.repos import (  # noqa: E402
    DEFAULT_REPOS,
    export_table_csv,
    list_tables,
    repo_dir_valid,
)


def import_csv_chunked(
    csv_path: Path,
    table: str,
    schema: str,
    *,
    eng,
    chunksize: int,
) -> int:
    total = 0
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk.to_sql(
            table,
            eng,
            schema=schema,
            if_exists="replace" if first else "append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )
        total += len(chunk)
        first = False
    return total


def import_repo(
    repo_dir: Path,
    schema: str,
    *,
    eng,
    chunksize: int,
    tables: list[str] | None,
    dry_run: bool,
) -> dict[str, int]:
    if not repo_dir_valid(repo_dir):
        raise FileNotFoundError(f"Depot Dolt introuvable : {repo_dir} (pas de .dolt/)")

    repo_tables = list_tables(repo_dir)
    if tables:
        missing = sorted(set(tables) - set(repo_tables))
        if missing:
            raise ValueError(f"Tables absentes dans {repo_dir.name} : {', '.join(missing)}")
        repo_tables = [t for t in repo_tables if t in tables]

    if dry_run:
        print(f"  [dry-run] {repo_dir.name} -> schema {schema} : {len(repo_tables)} table(s)")
        for t in repo_tables:
            print(f"    - {t}")
        return {}

    ensure_schema(schema, eng=eng)
    imported: dict[str, int] = {}

    for table in repo_tables:
        print(f"  -> {schema}.{table} ...", flush=True)
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            prefix=f"dolt_{schema}_{table}_",
            delete=False,
        ) as tmp:
            csv_path = Path(tmp.name)

        try:
            export_table_csv(repo_dir, table, csv_path)
            if csv_path.stat().st_size == 0:
                print(f"     (vide, ignore)")
                continue
            rows = import_csv_chunked(
                csv_path,
                table,
                schema,
                eng=eng,
                chunksize=chunksize,
            )
            imported[table] = rows
            print(f"     {rows:,} ligne(s)")
        finally:
            csv_path.unlink(missing_ok=True)

    return imported


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importe 4 depots Dolt (stocks, options, rates, earnings) vers PostgreSQL.",
    )
    parser.add_argument(
        "--dolt-root",
        type=Path,
        default=Path("/data"),
        help="Dossier contenant les clones (defaut: /data dans ubuntu-box)",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        metavar="NAME",
        help="Sous-ensemble de depots (stocks, options, rates, earnings). Defaut: les 4.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        help="Limiter a certaines tables (applique a chaque depot selectionne).",
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

    repos = DEFAULT_REPOS
    if args.repos:
        wanted = set(args.repos)
        repos = [r for r in DEFAULT_REPOS if r["dir"] in wanted]
        unknown = wanted - {r["dir"] for r in repos}
        if unknown:
            print(f"Erreur : depots inconnus : {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1

    dolt_root = args.dolt_root.expanduser().resolve()
    if not dolt_root.is_dir():
        print(f"Erreur : dossier Dolt introuvable : {dolt_root}", file=sys.stderr)
        return 1

    eng = None
    if not args.dry_run:
        eng = engine()
        test_connection(eng=eng)
        print(f"PostgreSQL OK — import depuis {dolt_root}\n")

    summary: dict[str, dict[str, int]] = {}
    for spec in repos:
        repo_dir = dolt_root / spec["dir"]
        schema = spec["schema"]
        print(f"[{spec['dir']}]")
        try:
            summary[schema] = import_repo(
                repo_dir,
                schema,
                eng=eng,
                chunksize=args.chunksize,
                tables=args.tables,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  Erreur : {exc}", file=sys.stderr)
            return 1
        print()

    if not args.dry_run:
        print("Creation des index PostgreSQL (filtres par symbole)...")
        created = ensure_warehouse_indexes(eng=eng)
        if created:
            for name in created:
                print(f"  + {name}")
        else:
            print("  (deja a jour)")
        print("Termine.")
        for schema, tables in summary.items():
            total = sum(tables.values())
            print(f"  {schema}: {len(tables)} table(s), {total:,} ligne(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
