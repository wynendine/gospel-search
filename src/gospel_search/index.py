"""SQLite (metadata + FTS5) and a memory-mapped .npy of vectors.

sqlite-vec can't be used here — this Python's sqlite3 was built without
`enable_load_extension` — so vectors live in a plain float32 .npy addressed by
`chunks.vec_row`. At ~190k x 1024 a brute-force matmul runs in ~19 ms, which is
exact and needs no index tuning, so nothing is lost by the substitution.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .config import DB_PATH, EMBED_DIMS, EMBED_MODEL, INDEX, META_PATH, VECTORS_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    key        TEXT UNIQUE NOT NULL,
    kind       TEXT NOT NULL,
    title      TEXT NOT NULL,
    url        TEXT NOT NULL,
    speaker    TEXT NOT NULL DEFAULT '',
    role       TEXT NOT NULL DEFAULT '',
    date       TEXT NOT NULL DEFAULT '',
    volume     TEXT NOT NULL DEFAULT '',
    book       TEXT NOT NULL DEFAULT '',
    chapter    INTEGER,
    conference TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(id),
    vec_row      INTEGER,
    kind         TEXT NOT NULL,
    anchor       TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    citation     TEXT NOT NULL,
    url          TEXT NOT NULL,
    speaker      TEXT NOT NULL DEFAULT '',
    date         TEXT NOT NULL DEFAULT '',
    display_text TEXT NOT NULL,
    window_text  TEXT NOT NULL,
    embed_text   TEXT NOT NULL,
    embedded     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS chunks_doc      ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_embedded ON chunks(embedded);
CREATE INDEX IF NOT EXISTS chunks_kind     ON chunks(kind);
CREATE INDEX IF NOT EXISTS chunks_date     ON chunks(date);

-- rowid is kept equal to chunks.id, so a bm25 hit maps straight to a chunk
-- with no join.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    display_text,
    citation,
    speaker,
    tokenize = 'porter unicode61'
);
"""


def connect(path: Path = DB_PATH, *, readonly: bool = False) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


DOCUMENT_FIELDS = (
    "kind", "title", "url", "speaker", "role", "date",
    "volume", "book", "chapter", "conference",
)


def add_document(conn: sqlite3.Connection, document, chunks) -> int:
    """Upsert a document and its chunks, per chunk. Returns chunks embedded anew.

    Chunks are matched by anchor (paragraph id, verse number), and one whose
    embed_text is unchanged keeps its row id — and therefore its vector. That
    is what makes re-running `build` free, and what lets a document gain a
    chunk (adding chapter summaries, say) without re-embedding its siblings.
    Everything else about the chunk is refreshed in place, since display text
    and citations don't affect the vector.
    """
    row = conn.execute(
        "SELECT id FROM documents WHERE key = ?", (document.key,)
    ).fetchone()
    values = [getattr(document, field) for field in DOCUMENT_FIELDS]

    if row is None:
        doc_id = conn.execute(
            f"""INSERT INTO documents (key, {', '.join(DOCUMENT_FIELDS)})
                VALUES ({', '.join('?' * (len(DOCUMENT_FIELDS) + 1))})""",
            [document.key, *values],
        ).lastrowid
        existing: dict[str, sqlite3.Row] = {}
    else:
        doc_id = row["id"]
        conn.execute(
            f"""UPDATE documents
                SET {', '.join(f'{f} = ?' for f in DOCUMENT_FIELDS)}
                WHERE id = ?""",
            [*values, doc_id],
        )
        existing = {
            r["anchor"]: r
            for r in conn.execute(
                "SELECT id, anchor, embed_text FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        }

    incoming = {chunk.anchor for chunk in chunks}
    stale = [r["id"] for anchor, r in existing.items() if anchor not in incoming]
    refresh, insert = [], []

    for chunk in chunks:
        previous = existing.get(chunk.anchor)
        if previous is not None and previous["embed_text"] == chunk.embed_text:
            refresh.append(
                (
                    chunk.ordinal, chunk.citation, chunk.url, document.speaker,
                    document.date, chunk.display_text, chunk.window_text,
                    previous["id"],
                )
            )
        else:
            if previous is not None:
                stale.append(previous["id"])
            insert.append(chunk)

    if refresh:
        conn.executemany(
            """UPDATE chunks SET ordinal = ?, citation = ?, url = ?, speaker = ?,
                                 date = ?, display_text = ?, window_text = ?
               WHERE id = ?""",
            refresh,
        )
    if stale:
        conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in stale])
    if insert:
        conn.executemany(
            """INSERT INTO chunks
                 (doc_id, kind, anchor, ordinal, citation, url, speaker, date,
                  display_text, window_text, embed_text)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    doc_id, chunk.kind, chunk.anchor, chunk.ordinal, chunk.citation,
                    chunk.url, document.speaker, document.date, chunk.display_text,
                    chunk.window_text, chunk.embed_text,
                )
                for chunk in insert
            ],
        )

    return len(insert)


def vector_rows(conn: sqlite3.Connection) -> int:
    """Rows the vector array needs: MAX(id), not COUNT(*).

    Re-ingesting a document deletes its chunks and inserts new ones at the end,
    so ids can have gaps. Sizing by COUNT would leave the tail of the array
    unaddressable; a gap just costs 4 KB of zeros that `hydrate` drops anyway.
    """
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]


def assign_vector_rows(conn: sqlite3.Connection) -> int:
    """Pin each chunk to a stable vector row. Returns the array size needed."""
    conn.execute("UPDATE chunks SET vec_row = id - 1")
    conn.commit()
    return vector_rows(conn)


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        """INSERT INTO chunks_fts(rowid, display_text, citation, speaker)
           SELECT id, display_text, citation, speaker FROM chunks"""
    )
    conn.commit()


def open_vectors(rows: int, dims: int = EMBED_DIMS) -> np.memmap:
    """Open the vector store, growing it in place if the corpus got bigger.

    Growing must preserve what's there: chunks are marked `embedded` in SQLite,
    so silently reallocating would leave those rows as zeros with nothing left
    to notice it — vectors that are present, wrong, and never re-embedded.
    """
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not VECTORS_PATH.exists():
        return np.lib.format.open_memmap(
            VECTORS_PATH, mode="w+", dtype="float32", shape=(rows, dims)
        )

    existing = np.lib.format.open_memmap(VECTORS_PATH, mode="r")
    if existing.shape == (rows, dims):
        del existing
        return np.lib.format.open_memmap(VECTORS_PATH, mode="r+")

    if existing.shape[1] != dims:
        raise SystemExit(
            f"Vector store has {existing.shape[1]} dims but config wants {dims}. "
            "Delete data/index and rebuild."
        )

    keep = min(existing.shape[0], rows)
    carried = np.array(existing[:keep])  # copy before the file is replaced
    del existing

    grown = np.lib.format.open_memmap(
        VECTORS_PATH, mode="w+", dtype="float32", shape=(rows, dims)
    )
    grown[:keep] = carried
    grown.flush()
    return grown


def load_vectors() -> np.memmap:
    if not VECTORS_PATH.exists():
        raise SystemExit("No vector index yet — run `gospel build` first.")
    return np.lib.format.open_memmap(VECTORS_PATH, mode="r")


def compact(conn: sqlite3.Connection) -> tuple[int, int]:
    """Renumber chunk ids to 1..N and rewrite vectors in the same order.

    Replacing documents leaves holes in the id space, and vector rows are
    addressed by id, so the array keeps the holes too. This squeezes them out
    without re-embedding anything: existing vectors are carried across to their
    new rows. Returns (rows before, rows after).
    """
    before = vector_rows(conn)
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks ORDER BY id")]
    after = len(ids)
    if before == after:
        return before, after

    carried = None
    if VECTORS_PATH.exists():
        source = np.lib.format.open_memmap(VECTORS_PATH, mode="r")
        # vec_row is always id - 1 and the array was sized by MAX(id), so every
        # gather index is in range. If that ever stops holding, the IndexError
        # here is far better than silently misaligned vectors.
        carried = np.array(source[[i - 1 for i in ids]])
        del source

    # Two passes through negative ids: the first can't collide with rows that
    # still hold their old id, the second flips them back and re-derives
    # vec_row from the pre-update (still negative) value.
    conn.executescript(
        """
        CREATE TEMP TABLE id_map AS
            SELECT id AS old_id, ROW_NUMBER() OVER (ORDER BY id) AS new_id FROM chunks;
        UPDATE chunks SET id = -(SELECT new_id FROM id_map WHERE old_id = chunks.id);
        UPDATE chunks SET id = -id, vec_row = -id - 1;
        DROP TABLE id_map;
        """
    )
    conn.commit()

    if carried is not None:
        rewritten = np.lib.format.open_memmap(
            VECTORS_PATH, mode="w+", dtype="float32", shape=(after, carried.shape[1])
        )
        rewritten[:] = carried
        rewritten.flush()
        del rewritten

    rebuild_fts(conn)
    previous_isolation = conn.isolation_level
    conn.isolation_level = None  # VACUUM cannot run inside a transaction
    conn.execute("VACUUM")
    conn.isolation_level = previous_isolation
    return before, after


def write_meta(conn: sqlite3.Connection, rows: int) -> None:
    counts = {
        row["kind"]: row["n"]
        for row in conn.execute("SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind")
    }
    INDEX.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL,
                "dims": EMBED_DIMS,
                "chunks": rows,
                "by_kind": counts,
                "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            },
            indent=2,
        )
    )


def read_meta() -> dict:
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text())
