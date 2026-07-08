#!/usr/bin/env python3
"""Attend que le NameNode HDFS soit pret."""

from __future__ import annotations

import sys
import time

import requests


def main() -> int:
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from src.env import load_dotenv

    load_dotenv()

    import os

    host = os.getenv("HDFS_NAMENODE", "localhost")
    port = os.getenv("HDFS_WEB_PORT", "9870")
    url = f"http://{host}:{port}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus"
    timeout = int(os.getenv("HDFS_WAIT_SECONDS", "180"))

    print(f"Attente HDFS NameNode {host}:{port} (max {timeout}s)...")
    for elapsed in range(timeout):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and "active" in response.text.lower():
                print("OK NameNode actif.")
                return 0
        except requests.RequestException:
            pass
        time.sleep(1)
        if elapsed % 10 == 0 and elapsed:
            print(f"  ... {elapsed}s")

    print("Erreur : HDFS non disponible.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
