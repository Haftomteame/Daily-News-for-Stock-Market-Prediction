#!/usr/bin/env python3
"""Synchronise lakehouse/bronze/news_reddit* (local) vers HDFS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv  # noqa: E402

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload tables Bronze news_reddit* (local -> HDFS)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "lakehouse" / "bronze",
        help="Dossier bronze local",
    )
    args = parser.parse_args()

    import os

    os.environ["STORAGE_BACKEND"] = "hdfs"

    import importlib
    import src.storage.io as storage_io

    importlib.reload(storage_io)

    from src.storage.io import join_path, write_bytes  # noqa: E402

    if not args.source.is_dir():
        print(f"Dossier absent : {args.source}", file=sys.stderr)
        return 1

    uploaded = 0
    for table_dir in sorted(args.source.iterdir()):
        if not table_dir.is_dir():
            continue
        if table_dir.name != "news_reddit" and not table_dir.name.startswith("news_reddit_"):
            continue
        for file_path in table_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(PROJECT_ROOT).as_posix()
            target = join_path(relative)
            write_bytes(file_path.read_bytes(), target)
            uploaded += 1
            print(f"  {relative} -> {target}")

    if uploaded == 0:
        print("Aucun fichier news_reddit* a synchroniser.", file=sys.stderr)
        return 1

    print(f"\nOK {uploaded} fichiers Reddit synchronises vers HDFS")
    print("UI HDFS : http://localhost:9870 -> /datax/lakehouse/bronze/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
