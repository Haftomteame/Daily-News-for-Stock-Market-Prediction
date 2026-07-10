#!/usr/bin/env python3
"""Crée les index PostgreSQL pour accélérer le dashboard (filtres par symbole)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.postgres import ensure_warehouse_indexes, pg_enabled, test_connection  # noqa: E402


def main() -> int:
    if not pg_enabled():
        print("Erreur : PGHOST et PGPASSWORD requis.", file=sys.stderr)
        return 1
    test_connection()
    print("Creation des index warehouse...")
    created = ensure_warehouse_indexes()
    if created:
        for name in created:
            print(f"  + {name}")
    else:
        print("  (deja a jour)")
    print("Termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
