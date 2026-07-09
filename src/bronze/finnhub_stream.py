"""Streaming Finnhub WebSocket (trades) -> bougies OHLCV."""

from __future__ import annotations

import csv
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from src.bronze.storage import append_bronze_rows
from src.config import BRONZE_TABLES, FINNHUB_BUCKET_MODE

FINNHUB_TOKEN_ENV = "FINNHUB_TOKEN"
FINNHUB_WS_URL = "wss://ws.finnhub.io"
NY_TZ = ZoneInfo("America/New_York")
BucketMode = Literal["minute", "day"]

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
DEFAULT_BRONZE_TABLE = BRONZE_TABLES.get("stock_prices_1m", "stock_prices_1m")


class FinnhubStreamError(Exception):
    """Erreur credentials ou connexion Finnhub WebSocket."""


def has_finnhub_token() -> bool:
    return bool(os.getenv(FINNHUB_TOKEN_ENV))


def _token() -> str:
    token = os.getenv(FINNHUB_TOKEN_ENV)
    if not token:
        raise FinnhubStreamError(
            f"{FINNHUB_TOKEN_ENV} requis (https://finnhub.io/ > Dashboard > API Key)."
        )
    return token


def resolve_bucket_mode(mode: str | None = None) -> BucketMode:
    value = (mode or os.getenv("FINNHUB_BUCKET_MODE", FINNHUB_BUCKET_MODE)).lower()
    if value in {"day", "daily", "1d", "d"}:
        return "day"
    return "minute"


def bronze_table_for_mode(mode: BucketMode) -> str:
    return BRONZE_TABLES["stock_prices"] if mode == "day" else BRONZE_TABLES["stock_prices_1m"]


def format_candle_date(bucket_ms: int, *, daily: bool = False) -> str:
    dt = datetime.fromtimestamp(bucket_ms / 1000, tz=NY_TZ)
    if daily:
        return dt.strftime("%Y-%m-%d 00:00:00")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class CandleAgg:
    """Agregation OHLCV sur fenetres fixes (minute) ou par jour (NY)."""

    bucket_sec: int = 60
    bucket_mode: BucketMode = "minute"
    cur_bucket_ms: int | None = field(default=None, init=False)
    o: float | None = field(default=None, init=False)
    h: float | None = field(default=None, init=False)
    l: float | None = field(default=None, init=False)
    c: float | None = field(default=None, init=False)
    v: float = field(default=0.0, init=False)

    def bucket_start_ms(self, ts_ms: int) -> int:
        if self.bucket_mode == "day":
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=NY_TZ)
            day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            return int(day_start.timestamp() * 1000)
        bucket_ms = self.bucket_sec * 1000
        return (ts_ms // bucket_ms) * bucket_ms

    def _row_from_state(self) -> dict:
        if self.cur_bucket_ms is None or self.o is None:
            raise FinnhubStreamError("Aucune bougie en cours.")
        return {
            "Date": format_candle_date(self.cur_bucket_ms, daily=self.bucket_mode == "day"),
            "Open": self.o,
            "High": self.h,
            "Low": self.l,
            "Close": self.c,
            "Volume": self.v,
            "Adj Close": self.c,
        }

    def _start_bucket(self, bucket_ms: int, price: float, size: float) -> None:
        self.cur_bucket_ms = bucket_ms
        self.o = self.h = self.l = self.c = price
        self.v = float(size)

    def _finish(self) -> dict:
        row = self._row_from_state()
        self.cur_bucket_ms = None
        self.o = self.h = self.l = self.c = None
        self.v = 0.0
        return row

    def current_row(self) -> dict | None:
        if self.cur_bucket_ms is None:
            return None
        return self._row_from_state()

    def update_trade(self, price: float, size: float, ts_ms: int) -> dict | None:
        bucket_ms = self.bucket_start_ms(ts_ms)
        if self.cur_bucket_ms is None:
            self._start_bucket(bucket_ms, price, size)
            return None

        if bucket_ms != self.cur_bucket_ms:
            finished = self._finish()
            self._start_bucket(bucket_ms, price, size)
            return finished

        assert self.h is not None and self.l is not None
        self.h = max(self.h, price)
        self.l = min(self.l, price)
        self.c = price
        self.v += float(size)
        return None

    def flush_partial(self) -> dict | None:
        if self.cur_bucket_ms is None:
            return None
        return self._finish()


class CandleSink(Protocol):
    def write(self, row: dict) -> None: ...

    def flush(self) -> None: ...


class CsvCandleSink:
    """Append incremental des bougies terminees vers un CSV."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        new_file = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OHLCV_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def flush(self) -> None:
        return None


class BronzeLakehouseSink:
    """Buffer puis flush vers lakehouse/bronze/."""

    def __init__(
        self,
        *,
        table: str = DEFAULT_BRONZE_TABLE,
        source_label: str,
        batch_id: str,
        flush_every: int = 1,
    ) -> None:
        self.table = table
        self.source_label = source_label
        self.batch_id = batch_id
        self.flush_every = max(1, flush_every)
        self._buffer: list[dict] = []

    def write(self, row: dict) -> None:
        self._buffer.append(row.copy())
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = append_bronze_rows(
            self._buffer,
            self.table,
            self.batch_id,
            self.source_label,
        )
        count = len(self._buffer)
        self._buffer.clear()
        print(f"[finnhub] {count} bougie(s) -> bronze/{self.table} ({path})", flush=True)


class MultiCandleSink:
    def __init__(self, sinks: list[CandleSink]) -> None:
        self.sinks = sinks

    def write(self, row: dict) -> None:
        for sink in self.sinks:
            sink.write(row)
        print(
            f"[finnhub] {row['Date']} O={row['Open']} H={row['High']} "
            f"L={row['Low']} C={row['Close']} V={row['Volume']}",
            flush=True,
        )

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()


class FinnhubOhlcvStreamer:
    """Client WebSocket Finnhub : trades -> OHLCV minute ou journalier."""

    def __init__(
        self,
        symbol: str,
        *,
        bucket_sec: int = 60,
        bucket_mode: BucketMode = "minute",
        sink: CandleSink,
        max_candles: int | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.bucket_mode = bucket_mode
        self.agg = CandleAgg(bucket_sec=bucket_sec, bucket_mode=bucket_mode)
        self.sink = sink
        self.max_candles = max_candles
        self._candles_written = 0
        self._stop = False
        self._token = _token()
        self._ws = None

    def _emit_candle(self, row: dict, *, count: bool = True) -> None:
        self.sink.write(row)
        if not count:
            return
        self._candles_written += 1
        if self.max_candles is not None and self._candles_written >= self.max_candles:
            self._stop = True

    def _on_trade_message(self, message: str) -> None:
        payload = json.loads(message)
        if payload.get("type") != "trade":
            return

        for trade in payload.get("data") or []:
            finished = self.agg.update_trade(
                float(trade["p"]),
                float(trade.get("v", 0.0)),
                int(trade["t"]),
            )
            if finished is not None:
                self._emit_candle(finished)
            elif self.bucket_mode == "day":
                current = self.agg.current_row()
                if current is not None:
                    self._emit_candle(current, count=False)

    def _shutdown(self) -> None:
        self._stop = True
        partial = self.agg.flush_partial()
        if partial is not None:
            self._emit_candle(partial)
        self.sink.flush()
        if self._ws is not None:
            self._ws.close()

    def run(self) -> int:
        try:
            import websocket
        except ImportError as exc:
            raise FinnhubStreamError(
                "Paquet websocket-client requis : pip install websocket-client"
            ) from exc

        url = f"{FINNHUB_WS_URL}?token={self._token}"
        backoff = 1

        def on_open(ws) -> None:
            label = "journalier" if self.bucket_mode == "day" else f"{self.agg.bucket_sec}s"
            print(f"[finnhub] connecte — abonnement {self.symbol} ({label})", flush=True)
            ws.send(json.dumps({"type": "subscribe", "symbol": self.symbol}))

        def on_message(ws, message: str) -> None:
            self._on_trade_message(message)
            if self._stop:
                ws.close()

        def on_error(_ws, error) -> None:
            if error is not None:
                print(f"[finnhub] erreur WS : {error}", flush=True)

        def handle_signal(_signum, _frame) -> None:
            print("[finnhub] arret demande — flush en cours...", flush=True)
            self._shutdown()

        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)

        while not self._stop:
            self._ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
            )
            self._ws.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop:
                break
            print(f"[finnhub] reconnexion dans {backoff}s...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

        return self._candles_written


def build_streamer(
    symbol: str,
    *,
    bucket_sec: int = 60,
    bucket_mode: str | None = None,
    out_csv: Path | None = None,
    lakehouse: bool = True,
    flush_every: int = 1,
    max_candles: int | None = None,
) -> FinnhubOhlcvStreamer:
    mode = resolve_bucket_mode(bucket_mode)
    sinks: list[CandleSink] = []
    batch_id = f"finnhub-stream-{uuid.uuid4().hex[:8]}"
    suffix = "1d" if mode == "day" else "1m"
    source_label = f"finnhub_ws_{symbol.lower()}_{suffix}"
    bronze_table = bronze_table_for_mode(mode)

    if lakehouse:
        sinks.append(
            BronzeLakehouseSink(
                table=bronze_table,
                source_label=source_label,
                batch_id=batch_id,
                flush_every=flush_every,
            )
        )
    if out_csv is not None:
        sinks.append(CsvCandleSink(out_csv))
    if not sinks:
        raise FinnhubStreamError("Au moins une sortie requise (--lakehouse et/ou --csv).")

    sink: CandleSink = sinks[0] if len(sinks) == 1 else MultiCandleSink(sinks)
    return FinnhubOhlcvStreamer(
        symbol,
        bucket_sec=bucket_sec,
        bucket_mode=mode,
        sink=sink,
        max_candles=max_candles,
    )
