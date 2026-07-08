#!/usr/bin/env python3
"""Copie lakehouse/ + monitoring/ local vers HDFS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_dotenv  # noqa: E402

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload lakehouse local -> HDFS")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT,
        help="Racine contenant lakehouse/ et monitoring/ (defaut: projet)",
    )
    args = parser.parse_args()

    import os

    os.environ["STORAGE_BACKEND"] = "hdfs"

    import importlib
    import src.storage.io as storage_io

    importlib.reload(storage_io)

    from src.storage.io import join_path, write_bytes  # noqa: E402

    uploaded = 0
    for folder in ("lakehouse", "monitoring"):
        root = args.source / folder
        if not root.exists():
            print(f"WARN dossier absent : {root}")
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(args.source).as_posix()
            target = join_path(relative)
            write_bytes(file_path.read_bytes(), target)
            uploaded += 1
            print(f"  {relative} -> {target}")

    if uploaded == 0:
        print("Aucun fichier a uploader.", file=sys.stderr)
        return 1

    print(f"\nOK {uploaded} fichiers uploades vers HDFS")
    print("Configurez .env : STORAGE_BACKEND=hdfs puis lancez le pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
