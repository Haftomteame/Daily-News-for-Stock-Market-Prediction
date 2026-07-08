"""Abstraction stockage local / HDFS (fsspec)."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import fsspec
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def storage_backend() -> str:
    return os.getenv("STORAGE_BACKEND", "local").lower()


def is_hdfs() -> bool:
    return storage_backend() == "hdfs"


def storage_root() -> str:
    if is_hdfs():
        return os.getenv("HDFS_BASE_PATH", "/datax").rstrip("/")
    return os.getenv("LOCAL_STORAGE_ROOT", str(PROJECT_ROOT)).rstrip("/\\")


def hdfs_namenode() -> str:
    host = os.getenv("HDFS_NAMENODE", "localhost")
    port = os.getenv("HDFS_PORT", "8020")
    return f"hdfs://{host}:{port}"


def get_fs():
    if is_hdfs():
        return fsspec.filesystem(
            "hdfs",
            host=os.getenv("HDFS_NAMENODE", "localhost"),
            port=int(os.getenv("HDFS_PORT", "8020")),
            user=os.getenv("HDFS_USER", "hdfs"),
        )
    return fsspec.filesystem("file")


def join_path(*parts: str) -> str:
    cleaned = [p.strip("/\\") for p in parts if p]
    root = storage_root()
    if is_hdfs():
        return "/" + "/".join([root.strip("/"), *cleaned])
    return str(Path(root).joinpath(*cleaned))


def lakehouse_parquet(layer: str, table: str) -> str:
    return join_path("lakehouse", layer, table, "data.parquet")


def lakehouse_dir(*parts: str) -> str:
    return join_path("lakehouse", *parts)


def monitoring_path(*parts: str) -> str:
    return join_path("monitoring", *parts)


def ml_file(name: str) -> str:
    return join_path("lakehouse", "ml", name)


def storage_label() -> str:
    if is_hdfs():
        return f"{hdfs_namenode()}{storage_root()}/lakehouse/"
    return join_path("lakehouse") + ("" if join_path("lakehouse").endswith("/") else "/")


def exists(path: str) -> bool:
    return get_fs().exists(path)


def makedirs(path: str) -> None:
    parent = path.replace("\\", "/").rsplit("/", 1)[0]
    if parent:
        get_fs().makedirs(parent, exist_ok=True)


def modified_time(path: str) -> float:
    fs = get_fs()
    if not fs.exists(path):
        return 0.0
    mtime = fs.modified(path)
    if isinstance(mtime, datetime):
        return mtime.timestamp()
    return float(mtime)


def file_size(path: str) -> int:
    fs = get_fs()
    if not fs.exists(path):
        return 0
    return int(fs.size(path))


def dir_size(path: str) -> int:
    fs = get_fs()
    if not fs.exists(path):
        return 0
    total = 0
    try:
        for entry in fs.find(path):
            if not fs.isdir(entry):
                total += int(fs.size(entry))
    except FileNotFoundError:
        return 0
    return total


def glob_paths(pattern: str) -> list[str]:
    return sorted(get_fs().glob(pattern))


def read_parquet(path: str) -> pd.DataFrame:
    with get_fs().open(path, "rb") as handle:
        return pd.read_parquet(handle)


def write_parquet(df: pd.DataFrame, path: str) -> str:
    makedirs(path)
    with get_fs().open(path, "wb") as handle:
        df.to_parquet(handle, index=False, engine="pyarrow")
    return path


def read_bytes(path: str) -> bytes:
    with get_fs().open(path, "rb") as handle:
        return handle.read()


def write_bytes(data: bytes, path: str) -> str:
    makedirs(path)
    with get_fs().open(path, "wb") as handle:
        handle.write(data)
    return path


def read_text(path: str, encoding: str = "utf-8") -> str:
    with get_fs().open(path, "r", encoding=encoding) as handle:
        return handle.read()


def write_text(text: str, path: str, encoding: str = "utf-8") -> str:
    makedirs(path)
    with get_fs().open(path, "w", encoding=encoding) as handle:
        handle.write(text)
    return path


def read_json(path: str) -> dict[str, Any]:
    return json.loads(read_text(path))


def write_json(data: dict[str, Any], path: str) -> str:
    write_text(json.dumps(data, indent=2, ensure_ascii=False), path)
    return path


def write_joblib(obj: Any, path: str) -> str:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    write_bytes(buffer.getvalue(), path)
    return path


def read_joblib(path: str) -> Any:
    return joblib.load(io.BytesIO(read_bytes(path)))


def query_duckdb(sql: str, tables: dict[str, str]) -> pd.DataFrame:
    """Execute SQL DuckDB en chargeant les Parquet via fsspec (local ou HDFS)."""
    import duckdb

    con = duckdb.connect()
    try:
        for alias, path in tables.items():
            con.register(alias, read_parquet(path))
        return con.execute(sql).df()
    finally:
        con.close()
