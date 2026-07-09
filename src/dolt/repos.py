"""Lecture et export des depots Dolt clones."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Mapping depot clone -> schema PostgreSQL (dolt clone <org>/<name> cree le dossier <name>).
DEFAULT_REPOS: list[dict[str, str]] = [
    {"dir": "stocks", "schema": "stocks"},
    {"dir": "options", "schema": "options"},
    {"dir": "rates", "schema": "rates"},
    {"dir": "earnings", "schema": "earnings"},
]


def run_dolt(repo_dir: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["dolt", *args],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"dolt {' '.join(args)} a echoue dans {repo_dir}:\n{exc.output}"
        ) from exc


def repo_dir_valid(repo_dir: Path) -> bool:
    return (repo_dir / ".dolt").is_dir()


def list_tables(repo_dir: Path) -> list[str]:
    # Dolt >= 2.x : `dolt ls` ; versions plus anciennes : `dolt table ls`
    for args in (("ls",), ("table", "ls")):
        try:
            out = run_dolt(repo_dir, *args)
            return [
                line.strip()
                for line in out.splitlines()
                if line.strip() and not line.strip().startswith("Tables in")
            ]
        except RuntimeError as exc:
            if "Unknown Command" not in str(exc):
                raise
    raise RuntimeError(f"Impossible de lister les tables dans {repo_dir}")


def export_table_csv(repo_dir: Path, table: str, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_dolt(repo_dir, "table", "export", "-f", table, str(csv_path))
