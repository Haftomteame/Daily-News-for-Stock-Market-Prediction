"""Import Dolt -> PostgreSQL (un schema par depot)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.db.postgres import ensure_schema
from src.dolt.repos import export_table_csv, list_tables, repo_dir_valid


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


def import_repo_to_postgres(
    *,
    repo_dir: Path,
    schema: str,
    eng,
    chunksize: int = 5000,
    tables: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    if not repo_dir_valid(repo_dir):
        raise FileNotFoundError(f"Depot Dolt introuvable : {repo_dir} (pas de .dolt/)")

    repo_tables = list_tables(repo_dir)
    if tables:
        missing = sorted(set(tables) - set(repo_tables))
        if missing:
            raise ValueError(
                f"Tables absentes dans {repo_dir.name} : {', '.join(missing)}"
            )
        repo_tables = [t for t in repo_tables if t in tables]

    if dry_run:
        return {t: 0 for t in repo_tables}

    ensure_schema(schema, eng=eng)
    imported: dict[str, int] = {}

    for table in repo_tables:
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            prefix=f"dolt_{schema}_{table}_",
            delete=False,
        ) as tmp:
            csv_path = Path(tmp.name)

        try:
            export_table_csv(repo_dir, table, csv_path)
            if csv_path.stat().st_size == 0:
                continue
            rows = import_csv_chunked(
                csv_path,
                table,
                schema,
                eng=eng,
                chunksize=chunksize,
            )
            imported[table] = rows
        finally:
            csv_path.unlink(missing_ok=True)

    return imported

