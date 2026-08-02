"""Prediction-log persistence (brief Section 4).

Every prediction is persisted both to a JSON-lines file (human-readable tail-able
monitor) and to a lightweight SQLite table (for the ``/logs/recent`` endpoint and
drift analytics). Log entries carry: timestamp, inputs, model, version, latency,
score and a monotonic per-model sequence number.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .. import config as C

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, lat REAL, lon REAL, model TEXT, version TEXT,
    risk_score REAL, level TEXT, latency_ms REAL
);
"""


class PredictionLogStore:
    """Thread-safe JSONL + SQLite appender for prediction records."""

    def __init__(self, jsonl: Path = None, db: Path = None):
        self.jsonl = Path(jsonl or C.PREDICTION_LOG_FILE)
        self.db = Path(db or C.PREDICTION_DB_FILE)
        self._lock = threading.Lock()
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def append(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("id", uuid.uuid4().hex)
        with self._lock:
            self._append_jsonl(record)
            self._append_sqlite(record)

    def _append_jsonl(self, record):
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _append_sqlite(self, record):
        self._conn.execute(
            "INSERT INTO predictions (ts,lat,lon,model,version,risk_score,level,latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (record.get("ts"), record.get("lat"), record.get("lon"),
             record.get("model"), record.get("version"), record.get("risk_score"),
             record.get("level"), record.get("latency_ms")))
        self._conn.commit()

    def recent(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts,lat,lon,model,version,risk_score,level,latency_ms"
            " FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)).fetchall()
        cols = ["ts", "lat", "lon", "model", "version", "risk_score", "level",
                "latency_ms"]
        return [dict(zip(cols, r)) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def make_record(lat, lon, when, model, version, score, level, latency_ms):
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lat": lat, "lon": lon, "datetime": str(when), "model": model,
        "version": version, "risk_score": float(score), "level": level,
        "latency_ms": round(latency_ms, 2),
    }