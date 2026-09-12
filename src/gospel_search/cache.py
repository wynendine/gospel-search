"""Cache answered queries, so asking the same thing twice is free.

A search costs real money in Claude calls, and studying is repetitive: you ask,
read, rephrase slightly, come back to it tomorrow. Re-asking something you have
already asked should not spend anything.

Keyed on the query, the filters, and the pipeline flags together, because they
all change the answer. Entries are invalidated by the index's own chunk count
and embedding model, so rebuilding the corpus retires the cache automatically
instead of serving answers about a corpus that no longer exists.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass

from .config import INDEX

CACHE_PATH = INDEX / "query_cache.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key        TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    corpus     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_created ON answers(created_at);
"""


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def corpus_signature() -> str:
    """Identifies the index a cached answer was produced from."""
    from . import index as idx

    meta = idx.read_meta()
    return f"{meta.get('embed_model')}:{meta.get('dims')}:{meta.get('chunks')}"


def key(query: str, filters, flags: dict) -> str:
    material = json.dumps(
        {
            "q": query.strip().lower(),
            "f": asdict(filters) if is_dataclass(filters) else filters,
            "x": flags,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def get(cache_key: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT payload, corpus, created_at FROM answers WHERE key = ?", (cache_key,)
    ).fetchone()
    conn.close()
    if row is None or row["corpus"] != corpus_signature():
        return None
    payload = json.loads(row["payload"])
    payload["_cached_at"] = row["created_at"]
    return payload


def put(cache_key: str, query: str, payload: dict) -> None:
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO answers (key, query, payload, corpus, created_at)
           VALUES (?,?,?,?,?)""",
        (cache_key, query, json.dumps(payload), corpus_signature(), time.time()),
    )
    conn.commit()
    conn.close()


def stats() -> dict:
    if not CACHE_PATH.exists():
        return {"entries": 0, "bytes": 0, "stale": 0}
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    stale = conn.execute(
        "SELECT COUNT(*) FROM answers WHERE corpus <> ?", (corpus_signature(),)
    ).fetchone()[0]
    conn.close()
    return {"entries": total, "bytes": CACHE_PATH.stat().st_size, "stale": stale}


def clear(*, stale_only: bool = False) -> int:
    if not CACHE_PATH.exists():
        return 0
    conn = _connect()
    if stale_only:
        n = conn.execute(
            "DELETE FROM answers WHERE corpus <> ?", (corpus_signature(),)
        ).rowcount
    else:
        n = conn.execute("DELETE FROM answers").rowcount
    conn.commit()
    conn.close()
    return n
