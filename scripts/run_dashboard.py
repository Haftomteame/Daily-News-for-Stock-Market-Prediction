#!/usr/bin/env python3
"""Lance le dashboard Streamlit sur un port dedie."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8502


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    print(f"Dashboard: http://localhost:{port}")
    print("Arret: Ctrl+C")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
