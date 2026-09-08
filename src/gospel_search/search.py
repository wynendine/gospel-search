"""Hybrid retrieval: HyDE -> (dense + BM25) -> RRF -> Claude rerank.

Keeping both retrieval halves matters. Dense search alone fumbles exact names
and quoted phrases; BM25 alone is the keyword guessing this tool exists to
replace. Fusing them by rank covers both failure modes, and the rerank pass is
what turns "topically nearby" into "this is the passage you meant".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic
import numpy as np

from .config import (
    ANSWER_K,
    DENSE_K,
    HYDE_MODEL,
    LEXICAL_K,
    RERANK_K,
    RERANK_MODEL,
    RRF_K,
    anthropic_key,
)
from . import embed
from . import index as idx

_anthropic: anthropic.Anthropic | None = None


def claude() -> anthropic.Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic(api_key=anthropic_key())
    return _anthropic


@dataclass
class Filters:
    speaker: str | None = None
    after: int | None = None
    before: int | None = None
    source: str | None = None  # "talks" | "scriptures"

    def sql(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.speaker:
            clauses.append("c.speaker LIKE ?")
            params.append(f"%{self.speaker}%")
        if self.after:
            clauses.append("c.date >= ?")
            params.append(f"{self.after}-01-01")
        if self.before:
            clauses.append("c.date <= ?")
            params.append(f"{self.before}-12-31")
        if self.source == "talks":
            clauses.append("c.kind = 'talk'")
        elif self.source == "scriptures":
            clauses.append("c.kind IN ('verse', 'summary')")
        return (" AND ".join(clauses) if clauses else "1=1"), params

    @property
    def active(self) -> bool:
        return any([self.speaker, self.after, self.before, self.source])


@dataclass
class Result:
    chunk_id: int
    citation: str
    url: str
    speaker: str
    date: str
    kind: str
    title: str
    display_text: str
    window_text: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused: float = 0.0
    rerank: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


# --- Query expansion -------------------------------------------------------

HYDE_PROMPT = """You help search a corpus of General Conference talks and the \
standard works (Bible, Book of Mormon, Doctrine and Covenants, Pearl of Great Price).

The user is trying to find a passage they half-remember. Write 2-3 sentences of \
the passage they are most likely looking for, in the voice and register of the \
source — a conference talk or scripture, as fits the query. Do not answer the \
question, do not hedge, do not mention the search. Just write the passage as it \
would plausibly appear.

Query: {query}"""


def hyde(query: str) -> str:
    """A hypothetical passage, embedded alongside the query to sharpen recall."""
    response = claude().messages.create(
        model=HYDE_MODEL,
        max_tokens=400,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Retrieval halves ------------------------------------------------------


def dense(conn, vectors, query_vector, filters: Filters, k: int = DENSE_K):
    scores = vectors @ query_vector

    if filters.active:
        where, params = filters.sql()
        allowed = np.fromiter(
            (
                row[0]
                for row in conn.execute(
                    # vec_row is NULL only for chunks added since the last
                    # build finalized; they have no vector to match against.
                    f"SELECT c.vec_row FROM chunks c "
                    f"WHERE {where} AND c.vec_row IS NOT NULL",
                    params,
                )
            ),
            dtype=np.int64,
        )
        allowed = allowed[allowed < scores.shape[0]]
        if allowed.size == 0:
            return []
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[allowed] = True
        scores = np.where(mask, scores, -np.inf)

    k = min(k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(row) + 1, float(scores[row])) for row in top if np.isfinite(scores[row])]


_TOKEN = re.compile(r"[A-Za-z0-9']+")


def fts_query(query: str, limit: int = 24) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Quoting each token keeps apostrophes and punctuation from being parsed as
    FTS operators, which would otherwise raise on perfectly ordinary queries.
    """
    tokens = [t for t in _TOKEN.findall(query) if len(t) > 1][:limit]
    return " OR ".join(f'"{t}"' for t in tokens)


def lexical(conn, query: str, filters: Filters, k: int = LEXICAL_K):
    match = fts_query(query)
    if not match:
        return []
    where, params = filters.sql()
    rows = conn.execute(
        f"""SELECT f.rowid AS id, bm25(chunks_fts, 4.0, 2.0, 1.0) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ? AND {where}
            ORDER BY score LIMIT ?""",
        [match, *params, k],
    ).fetchall()
    return [(row["id"], -row["score"]) for row in rows]  # bm25 is lower-is-better


# --- Fusion ----------------------------------------------------------------


def fuse(dense_hits, lexical_hits, k: int = RRF_K):
    """Reciprocal rank fusion — rank-based, so the two score scales never meet."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for label, hits in (("dense", dense_hits), ("lexical", lexical_hits)):
        for rank, (chunk_id, _) in enumerate(hits):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            ranks.setdefault(chunk_id, {})[label] = rank + 1

    order = sorted(scores, key=lambda cid: -scores[cid])
    return [(cid, scores[cid], ranks[cid]) for cid in order]


def hydrate(conn, fused) -> list[Result]:
    if not fused:
        return []
    ids = [cid for cid, _, _ in fused]
    placeholders = ",".join("?" * len(ids))
    rows = {
        row["id"]: row
        for row in conn.execute(
            f"""SELECT c.id, c.citation, c.url, c.speaker, c.date, c.kind,
                       c.display_text, c.window_text, d.title
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({placeholders})""",
            ids,
        )
    }

    results = []
    for chunk_id, score, rank in fused:
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append(
            Result(
                chunk_id=chunk_id,
                citation=row["citation"],
                url=row["url"],
                speaker=row["speaker"],
                date=row["date"],
                kind=row["kind"],
                title=row["title"],
                display_text=row["display_text"],
                window_text=row["window_text"],
                dense_rank=rank.get("dense"),
                lexical_rank=rank.get("lexical"),
                fused=score,
            )
        )
    return results


# --- Rerank ----------------------------------------------------------------

RERANK_PROMPT = """A person is searching General Conference talks and the \
standard works for a passage they half-remember. Score how well each numbered \
passage answers their query.

Query: {query}

Score 0-10:
  9-10  this is almost certainly the passage they mean
  6-8   directly on the subject, plausibly what they want
  3-5   related subject, probably not the passage
  0-2   not relevant

Judge the passage's own content. A passage is not more relevant because it \
repeats the query's words, and not less relevant because it uses none of them.

Passages:
{passages}"""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def rerank(query: str, results: list[Result], k: int = RERANK_K) -> list[Result]:
    candidates = results[:k]
    if not candidates:
        return results

    passages = "\n\n".join(
        f"[{i}] {r.citation}\n{r.display_text[:700]}" for i, r in enumerate(candidates)
    )
    response = claude().messages.create(
        model=RERANK_MODEL,
        max_tokens=4000,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": RERANK_SCHEMA},
        },
        messages=[
            {
                "role": "user",
                "content": RERANK_PROMPT.format(query=query, passages=passages),
            }
        ],
    )

    text = next(b.text for b in response.content if b.type == "text")
    scores = {
        item["id"]: float(item["score"]) for item in json.loads(text).get("scores", [])
    }
    for position, result in enumerate(candidates):
        result.rerank = scores.get(position, 0.0)

    rest = results[k:]
    candidates.sort(key=lambda r: -r.rerank)
    return candidates + rest


# --- Entry point -----------------------------------------------------------


def search(
    query: str,
    *,
    n: int = 10,
    filters: Filters | None = None,
    use_hyde: bool = True,
    use_rerank: bool = True,
    conn=None,
    vectors=None,
) -> list[Result]:
    filters = filters or Filters()
    conn = conn or idx.connect(readonly=True)
    vectors = idx.load_vectors() if vectors is None else vectors

    texts = [query]
    if use_hyde:
        try:
            hypothetical = hyde(query)
            if hypothetical:
                texts.append(hypothetical)
        except Exception as exc:  # noqa: BLE001 - HyDE is an optimization, not a gate
            print(f"  (hyde unavailable: {type(exc).__name__}; using the raw query)")

    embedded = embed.embed_texts(texts)
    query_vector = embedded.mean(axis=0)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)

    dense_hits = dense(conn, vectors, query_vector, filters)
    lexical_hits = lexical(conn, query, filters)
    results = hydrate(conn, fuse(dense_hits, lexical_hits))

    if use_rerank and results:
        results = rerank(query, results)

    return results[:n]


def context_for_answer(results: list[Result], k: int = ANSWER_K) -> list[Result]:
    return results[:k]
