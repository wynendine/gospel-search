"""Embed chunks with OpenAI, resumably.

Progress is tracked in SQLite (`chunks.embedded`), so a crash or a Ctrl-C costs
only the batch in flight. Vectors are L2-normalized on write, which makes
search a plain dot product later.
"""

from __future__ import annotations

import time

import numpy as np
from openai import OpenAI

from .config import EMBED_BATCH, EMBED_DIMS, EMBED_MODEL, openai_key
from . import index as idx
from . import usage

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=openai_key())
    return _client


def embed_texts(texts: list[str], *, model: str = EMBED_MODEL) -> np.ndarray:
    """Embed a list of texts, returning L2-normalized float32 vectors."""
    for attempt in range(5):
        try:
            response = client().embeddings.create(
                model=model, input=texts, dimensions=EMBED_DIMS
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            if attempt == 4:
                raise
            wait = 2**attempt
            print(f"  embedding retry {attempt + 1}/4 in {wait}s ({type(exc).__name__})")
            time.sleep(wait)

    usage.record_embedding(model, getattr(response.usage, 'total_tokens', 0) or 0)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def pending_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()[0]


def embed_pending(conn, *, batch_size: int = EMBED_BATCH, progress=None) -> int:
    """Embed every chunk not yet embedded. Returns how many were done."""
    total_rows = idx.vector_rows(conn)
    if total_rows == 0:
        return 0

    vectors = idx.open_vectors(total_rows)
    done = 0

    while True:
        rows = conn.execute(
            """SELECT id, vec_row, embed_text FROM chunks
               WHERE embedded = 0 ORDER BY id LIMIT ?""",
            (batch_size,),
        ).fetchall()
        if not rows:
            break

        embeddings = embed_texts([row["embed_text"] for row in rows])
        for row, vector in zip(rows, embeddings):
            vectors[row["vec_row"]] = vector

        vectors.flush()
        conn.executemany(
            "UPDATE chunks SET embedded = 1 WHERE id = ?", [(row["id"],) for row in rows]
        )
        conn.commit()

        done += len(rows)
        if progress is not None:
            progress(done)

    del vectors
    return done


def estimate_cost(conn) -> tuple[int, float]:
    """Rough token count and dollar cost for what's still pending."""
    total_chars = (
        conn.execute(
            "SELECT COALESCE(SUM(LENGTH(embed_text)), 0) FROM chunks WHERE embedded = 0"
        ).fetchone()[0]
        or 0
    )
    tokens = int(total_chars / 4)  # ~4 chars per token for English prose
    return tokens, tokens / 1_000_000 * 0.13  # text-embedding-3-large
