#!/usr/bin/env python3
"""Exporte chaque table du depot Dolt earnings vers CSV en parallele."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dolt.repos import export_table_csv, list_tables, repo_dir_valid  # noqa: E402

REPO_NAME = "earnings"


@dataclass(frozen=True)
class ExportJob:
    table: str
    output_path: Path


@dataclass
class ExportResult:
    table: str
    output_path: Path
    bytes_written: int = 0
    skipped_empty: bool = False
    error: str | None = None


def _repo_dir(dolt_root: Path) -> Path:
    repo_dir = dolt_root / REPO_NAME
    if not repo_dir_valid(repo_dir):
        raise FileNotFoundError(
            f"Depot Dolt earnings introuvable : {repo_dir} (pas de .dolt/)"
        )
    return repo_dir


def _collect_jobs(
    repo_dir: Path,
    output_dir: Path,
    *,
    tables: list[str] | None,
) -> list[ExportJob]:
    repo_tables = list_tables(repo_dir)
    if tables:
        missing = sorted(set(tables) - set(repo_tables))
        if missing:
            raise ValueError(f"Tables absentes dans earnings : {', '.join(missing)}")
        repo_tables = [t for t in repo_tables if t in tables]

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        ExportJob(table=table, output_path=output_dir / f"{table}.csv")
        for table in repo_tables
    ]


def _run_export(repo_dir: Path, job: ExportJob) -> ExportResult:
    result = ExportResult(table=job.table, output_path=job.output_path)
    try:
        export_table_csv(repo_dir, job.table, job.output_path)
        size = job.output_path.stat().st_size
        if size == 0:
            job.output_path.unlink(missing_ok=True)
            result.skipped_empty = True
            return result
        result.bytes_written = size
        return result
    except (OSError, RuntimeError) as exc:
        result.error = str(exc)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporte les tables du depot Dolt earnings vers CSV en parallele.",
    )
    parser.add_argument(
        "--dolt-root",
        type=Path,
        default=Path("/data"),
        help="Dossier contenant le clone earnings/ (defaut: /data dans ubuntu-box)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie (defaut: <dolt-root>/exports/earnings)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        help="Limiter a certaines tables du depot earnings.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Nombre d'exports simultanes (defaut: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lister les exports prevus sans ecrire de fichiers",
    )
    args = parser.parse_args()

    dolt_root = args.dolt_root.expanduser().resolve()
    if not dolt_root.is_dir():
        print(f"Erreur : dossier Dolt introuvable : {dolt_root}", file=sys.stderr)
        return 1

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dolt_root / "exports" / REPO_NAME
    )

    try:
        repo_dir = _repo_dir(dolt_root)
        jobs = _collect_jobs(repo_dir, output_dir, tables=args.tables)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    if not jobs:
        print("Aucune table a exporter dans earnings.")
        return 0

    if args.dry_run:
        print(f"[dry-run] {len(jobs)} export(s) earnings -> {output_dir}\n")
        for job in jobs:
            print(f"  {job.table} -> {job.output_path}")
        return 0

    workers = max(1, args.workers)
    print(
        f"Export de {len(jobs)} table(s) earnings vers {output_dir} "
        f"({workers} worker(s) en parallele)\n",
        flush=True,
    )

    started = time.monotonic()
    ok = 0
    empty = 0
    failed = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_export, repo_dir, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result.error:
                failed += 1
                print(f"  [ERREUR] {result.table} : {result.error}", file=sys.stderr, flush=True)
            elif result.skipped_empty:
                empty += 1
                print(f"  [vide]   {result.table}", flush=True)
            else:
                ok += 1
                total_bytes += result.bytes_written
                print(
                    f"  [ok]     {result.table} -> {result.output_path} "
                    f"({result.bytes_written:,} octets)",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    print(
        f"\nTermine en {elapsed:.1f}s : {ok} export(s), "
        f"{empty} vide(s), {failed} erreur(s), {total_bytes:,} octets.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
