"""Stream the local index straight into Neon Postgres.

Loads via binary COPY rather than an intermediate CSV: the text form of a
512-dim vector is ~9 bytes per dimension against 2 for halfvec, which turned a
250 MB transfer into a 1 GB one.

Three size decisions, all measured rather than assumed:

* Vectors are truncated 1024 -> 512 dims. text-embedding-3-large is
  matryoshka-trained, so a prefix is still a valid embedding once renormalized.
  Measured on the eval set: recall@10 identical, MRR within noise.
* `embed_text` is dropped — it existed only to be embedded, and that happened.
* `window_text` is dropped and reconstructed at query time from neighbouring
  chunks (same doc_id, adjacent ordinal). 143 MB of pure duplication.

Usage:
    DATABASE_URL=postgresql://... .venv/bin/python scripts/load_postgres.py
    ... --skip-index      load rows only, add indexes later
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gospel_search import index as idx  # noqa: E402

TARGET_DIMS = 512

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

CREATE TABLE documents (
    id         integer PRIMARY KEY,
    key        text UNIQUE NOT NULL,
    kind       text NOT NULL,
    title      text NOT NULL,
    url        text NOT NULL,
    speaker    text NOT NULL DEFAULT '',
    role       text NOT NULL DEFAULT '',
    date       text NOT NULL DEFAULT '',
    volume     text NOT NULL DEFAULT '',
    book       text NOT NULL DEFAULT '',
    chapter    integer,
    conference text NOT NULL DEFAULT ''
);

CREATE TABLE chunks (
    id           integer PRIMARY KEY,
    doc_id       integer NOT NULL REFERENCES documents(id),
    kind         text NOT NULL,
    anchor       text NOT NULL,
    ordinal      integer NOT NULL,
    citation     text NOT NULL,
    url          text NOT NULL,
    speaker      text NOT NULL DEFAULT '',
    date         text NOT NULL DEFAULT '',
    display_text text NOT NULL,
    embedding    halfvec({TARGET_DIMS}) NOT NULL
);
"""

# Reconstructing the surrounding paragraphs needs a fast (doc_id, ordinal)
# lookup. The tsvector is generated rather than stored as a column to keep the
# row narrow; the GIN index is what actually serves BM25-ish ranking.
INDEXES = """
CREATE INDEX chunks_doc_ordinal ON chunks (doc_id, ordinal);
CREATE INDEX chunks_kind        ON chunks (kind);
CREATE INDEX chunks_date        ON chunks (date);
CREATE INDEX chunks_speaker_trgm ON chunks (speaker);
CREATE INDEX chunks_fts ON chunks
    USING GIN (to_tsvector('english', display_text || ' ' || citation));
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")

    local = idx.connect(readonly=True)
    vectors = idx.load_vectors()
    print(f"source: {vectors.shape[0]:,} vectors x {vectors.shape[1]} dims")

    with psycopg.connect(dsn, autocommit=False) as pg:
        register_vector(pg)
        print("creating schema...")
        pg.execute(SCHEMA)
        pg.commit()

        docs = local.execute(
            """SELECT id, key, kind, title, url, speaker, role, date,
                      volume, book, chapter, conference
               FROM documents ORDER BY id"""
        ).fetchall()
        with pg.cursor().copy(
            "COPY documents (id, key, kind, title, url, speaker, role, date,"
            " volume, book, chapter, conference) FROM STDIN (FORMAT BINARY)"
        ) as copy:
            copy.set_types([
                "int4", "text", "text", "text", "text", "text", "text", "text",
                "text", "text", "int4", "text",
            ])
            for r in docs:
                copy.write_row([
                    r["id"], r["key"], r["kind"], r["title"], r["url"],
                    r["speaker"], r["role"], r["date"], r["volume"], r["book"],
                    r["chapter"], r["conference"],
                ])
        pg.commit()
        print(f"documents: {len(docs):,}")

        started = time.time()
        done = 0
        cursor = local.execute(
            """SELECT id, doc_id, vec_row, kind, anchor, ordinal, citation, url,
                      speaker, date, display_text
               FROM chunks ORDER BY id"""
        )
        with pg.cursor().copy(
            "COPY chunks (id, doc_id, kind, anchor, ordinal, citation, url,"
            " speaker, date, display_text, embedding) FROM STDIN (FORMAT BINARY)"
        ) as copy:
            copy.set_types([
                "int4", "int4", "text", "text", "int4", "text", "text",
                "text", "text", "text", "halfvec",
            ])
            while True:
                batch = cursor.fetchmany(args.batch)
                if not batch:
                    break
                for r in batch:
                    v = np.asarray(
                        vectors[r["vec_row"]][:TARGET_DIMS], dtype=np.float32
                    )
                    norm = float(np.linalg.norm(v))
                    if norm < 1e-9:
                        continue  # never embedded; nothing to search against
                    copy.write_row([
                        r["id"], r["doc_id"], r["kind"], r["anchor"], r["ordinal"],
                        r["citation"], r["url"], r["speaker"], r["date"],
                        r["display_text"], v / norm,
                    ])
                    done += 1
                rate = done / max(time.time() - started, 0.01)
                print(f"  chunks: {done:,} ({rate:,.0f}/s)", end="\r", flush=True)
        pg.commit()
        print(f"\nchunks: {done:,} in {time.time() - started:.0f}s")

        if not args.skip_index:
            print("building indexes (this is the slow part)...")
            pg.execute(INDEXES)
            pg.commit()

        pg.execute("ANALYZE documents; ANALYZE chunks;")
        pg.commit()

        size = pg.execute(
            "SELECT pg_size_pretty(pg_database_size(current_database()))"
        ).fetchone()[0]
        print(f"database size: {size}")


if __name__ == "__main__":
    main()
