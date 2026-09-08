"""The talk -> scripture citation graph.

Talks footnote the verses they quote, and those footnotes are structured links:
`/study/scriptures/bofm/alma/32?lang=eng&id=p27#p27`. Parsing them turns the
corpus into a graph and answers a question class search cannot: *which talks
cite Alma 32:21*, *what does Holland quote when he preaches faith*, *what are
the most-cited verses in conference*.

Footnotes only exist from about 1990 (nothing before, ~15 refs per talk after
2000), so older talks are covered by a second pass over the prose, where
references were written inline — "(D&C 64:9)". That pass is gated on the book
name resolving through the same alias table the reference parser uses, which is
what keeps "April 2013:15" from being read as scripture.
"""

from __future__ import annotations

import csv
import re
import urllib.parse
from dataclasses import dataclass

from .sources.scriptures import CSV_PATH

# href="/study/scriptures/{volume}/{book}/{chapter}?...&id=p4#p4"
HREF = re.compile(r'href="/study/scriptures/([a-z0-9-]+)/([a-z0-9-]+)/(\d+)([^"]*)"')
# Inline in prose: "(D&C 64:9)", "see Alma 32:21", "Moroni 10:3–5"
INLINE = re.compile(
    r"\b((?:[1-4]\s+)?[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][a-z]+){0,2})\s+(\d+):(\d+)(?:\s*[-‒-―]\s*(\d+))?"
)


@dataclass(frozen=True)
class Citation:
    book: str  # canonical title, e.g. "Alma"
    chapter: int
    verses: tuple[int, ...]  # empty means the whole chapter
    text: str  # as written, e.g. "Alma 32:27"
    origin: str  # "footnote" | "inline"


_slugs: dict[tuple[str, str], str] | None = None


def slug_map() -> dict[tuple[str, str], str]:
    """(volume_slug, book_slug) -> canonical book title, from the source CSV."""
    global _slugs
    if _slugs is None:
        table: dict[tuple[str, str], str] = {}
        if CSV_PATH.exists():
            with CSV_PATH.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    table[(row["volume_lds_url"], row["book_lds_url"])] = row["book_title"]
        _slugs = table
    return _slugs


def _verses(query: str) -> tuple[int, ...]:
    """Verse numbers from the `id` parameter: p4, p8-p9, p12%2Cp14."""
    match = re.search(r"[?&]id=([^&#]+)", query)
    if not match:
        return ()

    found: list[int] = []
    for part in urllib.parse.unquote(match.group(1)).split(","):
        span = re.match(r"p(\d+)(?:-p(\d+))?$", part.strip())
        if not span:
            continue
        start = int(span.group(1))
        end = int(span.group(2) or start)
        if end - start > 200:  # a malformed range should not explode
            end = start
        found.extend(range(start, end + 1))
    return tuple(sorted(set(found)))


def from_footnotes(payload: dict) -> list[Citation]:
    """Structured scripture links out of a talk's footnotes."""
    found: list[Citation] = []
    for note in (payload.get("content", {}).get("footnotes") or {}).values():
        text = (note.get("text") or "").replace("&amp;", "&")
        for volume, book, chapter, query in HREF.findall(text):
            title = slug_map().get((volume, book))
            if not title:
                continue
            verses = _verses(query)
            label = f"{title} {chapter}" + (f":{verses[0]}" if verses else "")
            found.append(
                Citation(title, int(chapter), verses, label, "footnote")
            )
    return found


def from_prose(text: str, aliases: dict[str, str]) -> list[Citation]:
    """References written inline, for talks that predate footnoting.

    `aliases` is the reference parser's normalized-name table; requiring a hit
    there is what keeps ordinary text with a colon and numbers out of the graph.
    """
    from .reference import _normalize

    found: list[Citation] = []
    for book, chapter, start, end in INLINE.findall(text):
        title = aliases.get(_normalize(book))
        if not title:
            continue
        first, last = int(start), int(end or start)
        if last < first or last - first > 200:
            last = first
        verses = tuple(range(first, last + 1))
        found.append(
            Citation(title, int(chapter), verses, f"{title} {chapter}:{start}", "inline")
        )
    return found


# --- Queries over the graph -------------------------------------------------


def citing_filter(filters) -> tuple[str, list]:
    """Filter clauses against the *citing talk*, aliased `d`.

    Filters describes a passage search, where speaker and date live on the
    chunk. Here those predicates belong to the talk doing the citing, and
    source/volume/book describe the scripture being cited rather than anything
    to filter the citing talk by — so only the two that transfer are used.
    Rewriting Filters.sql() with string substitution was the alternative, and
    that breaks the first time the clause shape changes.
    """
    if filters is None:
        return "1=1", []
    clauses, params = [], []
    if filters.speaker:
        clauses.append("d.speaker LIKE ?")
        params.append(f"%{filters.speaker}%")
    if filters.after:
        clauses.append("d.date >= ?")
        params.append(f"{filters.after}-01-01")
    if filters.before:
        clauses.append("d.date <= ?")
        params.append(f"{filters.before}-12-31")
    return (" AND ".join(clauses) if clauses else "1=1"), params


def citing_talks(conn, ref, filters=None, limit: int = 50):
    """Talks that cite a reference, newest first.

    A verse reference matches that verse; a chapter-only reference matches
    anything cited from the chapter, which is what someone asking about
    "Alma 32" almost always means.
    """
    where, params = citing_filter(filters)

    if ref.verse is None:
        sql = """
            SELECT DISTINCT d.id, d.speaker, d.title, d.date, d.url, ct.citation,
                            ct.src_chunk_id
              FROM citations ct
              JOIN documents d ON d.id = ct.src_doc_id
              JOIN documents t ON t.id = ct.tgt_doc_id
             WHERE t.book = ? AND t.chapter = ? AND {where}
             ORDER BY d.date DESC LIMIT ?"""
        args = [ref.book, ref.chapter, *params, limit]
    else:
        sql = """
            SELECT DISTINCT d.id, d.speaker, d.title, d.date, d.url, ct.citation,
                            ct.src_chunk_id
              FROM citations ct
              JOIN documents d ON d.id = ct.src_doc_id
              JOIN chunks c2 ON c2.id = ct.tgt_chunk_id
              JOIN documents t ON t.id = c2.doc_id
             WHERE t.book = ? AND t.chapter = ?
               AND CAST(c2.anchor AS INTEGER) BETWEEN ? AND ? AND {where}
             ORDER BY d.date DESC LIMIT ?"""
        args = [ref.book, ref.chapter, ref.verse, ref.end or ref.verse, *params, limit]

    return conn.execute(sql.format(where=where), args).fetchall()


def most_cited(conn, filters=None, limit: int = 20):
    """Verses cited by the most distinct talks."""
    where, params = citing_filter(filters)
    return conn.execute(
        f"""SELECT c2.citation, c2.display_text,
                   COUNT(DISTINCT ct.src_doc_id) AS talks
              FROM citations ct
              JOIN chunks c2 ON c2.id = ct.tgt_chunk_id
              JOIN documents d ON d.id = ct.src_doc_id
             WHERE {where}
             GROUP BY c2.citation, c2.display_text
             ORDER BY talks DESC LIMIT ?""",
        [*params, limit],
    ).fetchall()
