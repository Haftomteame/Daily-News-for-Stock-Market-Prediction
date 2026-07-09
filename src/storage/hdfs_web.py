"""Client WebHDFS (compatible Docker, sans libhdfs natif)."""

from __future__ import annotations

import fnmatch
import os
from typing import Iterator
from urllib.parse import urlparse, urlunparse

import requests
from hdfs import InsecureClient
from hdfs.util import HdfsError


def _rewrite_datanode_redirect(response: requests.Response, *args, **kwargs) -> requests.Response:
    """Reecrit les redirects WebHDFS vers un hote datanode joignable."""
    if response.status_code not in {301, 302, 303, 307, 308}:
        return response
    location = response.headers.get("Location")
    if not location or "/webhdfs/" not in location:
        return response
    parsed = urlparse(location)
    datanode_host = os.getenv("HDFS_DATANODE_HOST", "localhost")
    datanode_port = os.getenv("HDFS_DATANODE_WEB_PORT", "9864")
    target = f"{datanode_host}:{datanode_port}"
    # Depuis un conteneur Docker, le datanode annonce souvent localhost:9864.
    if parsed.hostname in {"localhost", "127.0.0.1"}:
        if parsed.netloc != target:
            response.headers["Location"] = urlunparse(parsed._replace(netloc=target))
        return response
    if parsed.hostname in {datanode_host, os.getenv("HDFS_NAMENODE", "localhost")}:
        return response
    response.headers["Location"] = urlunparse(parsed._replace(netloc=target))
    return response


class WebHdfsFilesystem:
    def __init__(self) -> None:
        host = os.getenv("HDFS_NAMENODE", "localhost")
        port = os.getenv("HDFS_WEB_PORT", "9870")
        user = os.getenv("HDFS_USER", "hdfs")
        session = requests.Session()
        session.hooks["response"].append(_rewrite_datanode_redirect)
        self._client = InsecureClient(f"http://{host}:{port}", user=user, session=session)

    @staticmethod
    def _path(path: str) -> str:
        return path.replace("\\", "/")

    def exists(self, path: str) -> bool:
        try:
            self._client.status(self._path(path))
            return True
        except HdfsError:
            return False

    def makedirs(self, path: str, exist_ok: bool = True, **_kwargs) -> None:
        hdfs_path = self._path(path)
        if self.exists(hdfs_path):
            if exist_ok:
                return
            raise FileExistsError(hdfs_path)
        parent = hdfs_path.rsplit("/", 1)[0]
        if parent and parent != "/":
            self.makedirs(parent)
        self._client.makedirs(hdfs_path)

    def open(self, path: str, mode: str = "rb", **kwargs):
        hdfs_path = self._path(path)
        # NOTE: hdfs.InsecureClient.read/write renvoie un context manager
        # compatible avec "with fs.open(...) as f:" attendu par src/storage/io.py.
        if mode == "rb":
            return self._client.read(hdfs_path)
        if mode == "r":
            encoding = kwargs.get("encoding", "utf-8")
            return self._client.read(hdfs_path, encoding=encoding)
        if mode == "wb":
            return self._client.write(hdfs_path, overwrite=True)
        if mode == "w":
            encoding = kwargs.get("encoding", "utf-8")
            return self._client.write(hdfs_path, overwrite=True, encoding=encoding)
        raise ValueError(f"Mode non supporte : {mode}")

    def modified(self, path: str) -> float:
        status = self._client.status(self._path(path))
        return float(status.get("modificationTime", 0)) / 1000.0

    def size(self, path: str) -> int:
        return int(self._client.status(self._path(path))["length"])

    def isdir(self, path: str) -> bool:
        return self._client.status(self._path(path))["type"] == "DIRECTORY"

    def glob(self, pattern: str) -> list[str]:
        pattern = self._path(pattern)
        if "*" not in pattern:
            return [pattern] if self.exists(pattern) else []
        base, _, tail = pattern.rpartition("/")
        base = base or "/"
        if not self.exists(base):
            return []
        matches: list[str] = []
        for entry in self._client.list(base, status=False):
            full = f"{base.rstrip('/')}/{entry}"
            if fnmatch.fnmatch(entry, tail):
                matches.append(full)
        return matches

    def find(self, path: str) -> Iterator[str]:
        hdfs_path = self._path(path)
        if not self.exists(hdfs_path):
            return iter(())

        def _walk(current: str) -> Iterator[str]:
            try:
                entries = self._client.list(current, status=True)
            except HdfsError:
                return
            for entry in entries:
                if isinstance(entry, tuple):
                    name, status = entry
                else:
                    name, status = entry, self._client.status(f"{current.rstrip('/')}/{entry}")
                full = name if str(name).startswith("/") else f"{current.rstrip('/')}/{name}"
                if status.get("type") == "DIRECTORY":
                    yield from _walk(full)
                else:
                    yield full

        return _walk(hdfs_path)


def hdfs_web_url() -> str:
    host = os.getenv("HDFS_NAMENODE", "localhost")
    port = os.getenv("HDFS_WEB_PORT", "9870")
    return f"http://{host}:{port}"
