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
    MAX_PER_DOC,
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
    volume: str | None = None  # "Book of Mormon", "Old Testament", ...
    book: str | None = None  # "Alma", "Moroni", ...

    def sql(self) -> tuple[str, list]:
        """WHERE fragment for the lexical half. Assumes `documents d` is joined."""
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
        if self.volume:
            clauses.append("d.volume LIKE ?")
            params.append(f"%{self.volume}%")
        if self.book:
            clauses.append("d.book LIKE ?")
            params.append(f"%{self.book}%")
        return (" AND ".join(clauses) if clauses else "1=1"), params

    @property
    def active(self) -> bool:
        return any(
            [self.speaker, self.after, self.before, self.source, self.volume, self.book]
        )


@dataclass
class Coverage:
    """What the answer actually got to look at.

    The failure mode of this tool is sounding authoritative while having read a
    thin slice — a comparison of two apostles drawn from one talk each reads
    exactly like a survey of forty. Reporting the slice is part of the answer.
    """

    passages: int = 0
    documents: int = 0
    candidates: int = 0  # size of the fused pool before truncation
    matching_documents: int | None = None  # documents the filter admits, if filtered

    def summary(self) -> str:
        parts = [f"{self.passages} passages from {self.documents} sources"]
        if self.matching_documents:
            parts.append(f"{self.matching_documents} matched the filter")
        parts.append(f"{self.candidates} candidates considered")
        return " · ".join(parts)


@dataclass
class Result:
    chunk_id: int
    doc_id: int
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
    facet: str = ""  # thematic: the sub-question that surfaced this passage

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


_columns: "Columns | None" = None

KIND_CODES = {"talk": 0, "verse": 1, "summary": 2}


def filter_columns(conn) -> "Columns":
    """Kind, year, document, and dictionary-encoded strings per vector row.

    Filtering used to pull every matching vec_row out of SQLite through a
    Python generator — for `--source talks` that is 146k integers crossing the
    interpreter boundary on every query, which made a *filtered* search 8x
    slower than an unfiltered one. These arrays cost ~950 KB and turn the
    common filters into vectorized comparisons.

    Speakers are dictionary-encoded: 640 distinct names over 146k chunks, so a
    substring match scans the 640 names and the result is an `isin` over codes,
    instead of a leading-wildcard LIKE that no index can serve.
    """
    global _columns
    if _columns is None:
        rows = max(conn.execute("SELECT MAX(vec_row) FROM chunks").fetchone()[0] or -1, -1) + 1
        kind = np.full(rows, -1, dtype=np.int8)
        year = np.zeros(rows, dtype=np.int16)
        doc = np.full(rows, -1, dtype=np.int32)

        # Dictionary-encoded string columns: (array of codes, list of names).
        encoded = {"speaker": {}, "volume": {}, "book": {}}
        arrays = {
            name: np.full(rows, -1, dtype=np.int16) for name in encoded
        }
        names = {name: [] for name in encoded}

        for vec_row, k, date, doc_id, speaker, volume, book in conn.execute(
            """SELECT c.vec_row, c.kind, c.date, c.doc_id, c.speaker, d.volume, d.book
                 FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.vec_row IS NOT NULL"""
        ):
            kind[vec_row] = KIND_CODES.get(k, -1)
            year[vec_row] = int(date[:4]) if date[:4].isdigit() else 0
            doc[vec_row] = doc_id
            for column, value in (
                ("speaker", speaker), ("volume", volume), ("book", book)
            ):
                if not value:
                    continue
                code = encoded[column].get(value)
                if code is None:
                    code = encoded[column][value] = len(names[column])
                    names[column].append(value)
                arrays[column][vec_row] = code

        _columns = Columns(kind, year, doc, arrays, names)
    return _columns


@dataclass
class Columns:
    kind: np.ndarray
    year: np.ndarray
    doc: np.ndarray
    codes: dict[str, np.ndarray]
    names: dict[str, list[str]]

    def matching(self, field: str, needle: str) -> np.ndarray | None:
        """Mask for a case-insensitive substring match on a dictionary column."""
        low = needle.lower()
        wanted = [i for i, n in enumerate(self.names[field]) if low in n.lower()]
        if not wanted:
            return None
        return np.isin(self.codes[field], np.array(wanted, dtype=np.int16))


def dense(conn, vectors, query_vector, filters: Filters, k: int = DENSE_K):
    scores = vectors @ query_vector

    if filters.active:
        cols = filter_columns(conn)
        mask = np.ones(scores.shape[0], dtype=bool)

        if filters.source == "talks":
            mask &= cols.kind == KIND_CODES["talk"]
        elif filters.source == "scriptures":
            mask &= (cols.kind == KIND_CODES["verse"]) | (
                cols.kind == KIND_CODES["summary"]
            )
        # Scripture rows carry year 0, so a date bound excludes them — which is
        # the right reading of "conference talks after 2010".
        if filters.after:
            mask &= cols.year >= filters.after
        if filters.before:
            mask &= (cols.year <= filters.before) & (cols.year > 0)

        for field, needle in (
            ("speaker", filters.speaker),
            ("volume", filters.volume),
            ("book", filters.book),
        ):
            if not needle:
                continue
            found = cols.matching(field, needle)
            if found is None:
                return []
            mask &= found

        if not mask.any():
            return []
        scores = np.where(mask, scores, -np.inf)

    k = min(k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(row) + 1, float(scores[row])) for row in top if np.isfinite(scores[row])]


_TOKEN = re.compile(r"[A-Za-z0-9']+")

# Deliberately a fixed English stopword list rather than a corpus-frequency
# cutoff: "God" and "Lord" appear in a huge share of this corpus and are highly
# meaningful, so frequency is the wrong signal here.
_STOPWORDS = frozenset(
    """a am an and are as at be been being but by did do does doing done for from
    had has have he her his i if in into is it its me my no nor not of on only or
    our shall she should so some such than that the their them then there these
    they this those to too under until up upon was we were what when where which
    while who whom why will with would you your""".split()
)


def fts_query(query: str, limit: int = 24) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Quoting each token keeps apostrophes and punctuation from being parsed as
    FTS operators, which would otherwise raise on perfectly ordinary queries.

    Stopwords are dropped because the terms are OR-ed: left in, a long question
    matches on "the", "his", and "will", and BM25's top hits share no content
    word with the query at all. On the eval set removing them moved hit@1 from
    45% to 59% and MRR from 0.621 to 0.703, with recall unchanged.
    """
    tokens = [t for t in _TOKEN.findall(query) if len(t) > 1]
    content = [t for t in tokens if t.lower() not in _STOPWORDS]
    # A query of nothing but stopwords still deserves its literal reading.
    return " OR ".join(f'"{t}"' for t in (content or tokens)[:limit])


def lexical(conn, query: str, filters: Filters, k: int = LEXICAL_K):
    match = fts_query(query)
    if not match:
        return []
    where, params = filters.sql()
    rows = conn.execute(
        f"""SELECT f.rowid AS id, bm25(chunks_fts, 4.0, 2.0, 1.0) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.doc_id
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
            f"""SELECT c.id, c.doc_id, c.citation, c.url, c.speaker, c.date, c.kind,
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
                doc_id=row["doc_id"],
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


def diversify(results: list[Result], max_per_doc: int, n: int) -> list[Result]:
    """Take the top n, but let no single document dominate.

    Without this a comparison of two speakers drew six of its eight passages
    from two talks — the ranking is per-chunk, and a talk that is strongly on
    topic wins every slot. Rank order is preserved; over-represented documents
    are skipped and backfilled from further down.
    """
    if max_per_doc <= 0:
        return results[:n]

    kept: list[Result] = []
    overflow: list[Result] = []
    seen: dict[int, int] = {}

    for r in results:
        if seen.get(r.doc_id, 0) < max_per_doc:
            seen[r.doc_id] = seen.get(r.doc_id, 0) + 1
            kept.append(r)
        else:
            overflow.append(r)
        if len(kept) == n:
            return kept

    # Not enough distinct documents to fill n — fall back to the best of what
    # was skipped rather than returning a short list.
    return (kept + overflow)[:n]


def coverage(
    conn, results: list[Result], candidates: int, filters: Filters
) -> Coverage:
    matching = None
    if filters.active:
        where, params = filters.sql()
        matching = conn.execute(
            f"""SELECT COUNT(DISTINCT c.doc_id) FROM chunks c
                JOIN documents d ON d.id = c.doc_id WHERE {where}""",
            params,
        ).fetchone()[0]
    return Coverage(
        passages=len(results),
        documents=len({r.doc_id for r in results}),
        candidates=candidates,
        matching_documents=matching,
    )


# --- Entry point -----------------------------------------------------------


def search_with_coverage(
    query: str,
    *,
    n: int = 10,
    filters: Filters | None = None,
    use_hyde: bool = True,
    use_rerank: bool = True,
    max_per_doc: int = MAX_PER_DOC,
    conn=None,
    vectors=None,
) -> tuple[list[Result], Coverage]:
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
    fused = fuse(dense_hits, lexical_hits)
    results = hydrate(conn, fused)

    if use_rerank and results:
        results = rerank(query, results)

    # Diversify after reranking, so relevance decides the order and the cap only
    # decides who gets crowded out.
    results = diversify(results, max_per_doc, n)
    return results, coverage(conn, results, len(fused), filters)


def search(query: str, **kwargs) -> list[Result]:
    """Results only, for callers that don't need the coverage report."""
    return search_with_coverage(query, **kwargs)[0]


def context_for_answer(results: list[Result], k: int = ANSWER_K) -> list[Result]:
    return results[:k]
