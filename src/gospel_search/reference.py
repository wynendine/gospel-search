"""Resolve a scripture reference to the exact verses it names.

Typing a reference used to go through semantic search, which is the wrong tool
for a question with one correct answer: "D&C 121:7" returned 121:40, "1 Ne 3:7"
returned 2 Nephi 7:10, and "Moroni 10:4" returned a 1975 talk. A reference is a
lookup, not a ranking.

Book names come from the corpus itself plus the official short titles already
in the source CSV ("D&C", "1 Ne.", "JS-H"), so the alias table is derived
rather than hand-maintained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Alma 32:21", "Alma 32:21-23", "D&C 121", "1 Ne. 3:7", "Moroni 10:3–5"
PATTERN = re.compile(
    r"""^\s*
    (?P<book>(?:[1-4]\s*)?[A-Za-z&.\-—\s]+?)  # book, optional leading number
    \s+
    (?P<chapter>\d+)
    (?:\s*[:.]\s*(?P<verse>\d+)
       (?:\s*[-‒-―]\s*(?P<end>\d+))?     # hyphen or any dash
    )?
    \s*$""",
    re.VERBOSE,
)


@dataclass
class Reference:
    book: str  # canonical book title, e.g. "Doctrine and Covenants"
    chapter: int
    verse: int | None = None
    end: int | None = None

    def citation(self) -> str:
        if self.verse is None:
            return f"{self.book} {self.chapter}"
        if self.end and self.end != self.verse:
            return f"{self.book} {self.chapter}:{self.verse}-{self.end}"
        return f"{self.book} {self.chapter}:{self.verse}"


def _normalize(text: str) -> str:
    """Fold case, punctuation, and spacing so '1 Ne.' == '1ne' == '1 nephi'."""
    return re.sub(r"[^a-z0-9&]", "", text.lower())


_aliases: dict[str, str] | None = None


def aliases(conn) -> dict[str, str]:
    """Normalized name/abbreviation -> canonical book title."""
    global _aliases
    if _aliases is not None:
        return _aliases

    table: dict[str, str] = {}
    for book, short in conn.execute(
        "SELECT DISTINCT book, title FROM documents WHERE kind='chapter' AND book<>''"
    ):
        table[_normalize(book)] = book

    # Official short titles live in the source CSV, not the index.
    from .sources.scriptures import CSV_PATH

    if CSV_PATH.exists():
        import csv as _csv

        with CSV_PATH.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                table.setdefault(_normalize(row["book_short_title"]), row["book_title"])
                table.setdefault(_normalize(row["book_title"]), row["book_title"])

    # A few spellings people actually type that no official list carries.
    extra = {
        "dc": "Doctrine and Covenants",
        "d&c": "Doctrine and Covenants",
        "doctrineandcovenants": "Doctrine and Covenants",
        "jsh": "Joseph Smith--History",
        "josephsmithhistory": "Joseph Smith--History",
        "jst": "Joseph Smith Translation",
        "aof": "Articles of Faith",
        "articlesoffaith": "Articles of Faith",
        "songofsolomon": "Song of Solomon",
        "wordsofmormon": "Words of Mormon",
    }
    for key, value in extra.items():
        if _normalize(value) in table or value in table.values():
            table.setdefault(key, value)

    _aliases = table
    return table


def parse(text: str, conn) -> Reference | None:
    """Parse a scripture reference, or None if the text isn't one."""
    match = PATTERN.match(text.strip())
    if not match:
        return None

    book = aliases(conn).get(_normalize(match.group("book")))
    if not book:
        return None

    verse = match.group("verse")
    end = match.group("end")
    return Reference(
        book=book,
        chapter=int(match.group("chapter")),
        verse=int(verse) if verse else None,
        end=int(end) if end else (int(verse) if verse else None),
    )


def lookup(conn, ref: Reference, limit: int = 60):
    """Chunk ids for a reference: the named verses, or a whole chapter."""
    if ref.verse is None:
        rows = conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE d.book = ? AND d.chapter = ? AND c.kind IN ('verse','summary')
               ORDER BY c.ordinal LIMIT ?""",
            (ref.book, ref.chapter, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE d.book = ? AND d.chapter = ? AND c.kind = 'verse'
                 AND CAST(c.anchor AS INTEGER) BETWEEN ? AND ?
               ORDER BY CAST(c.anchor AS INTEGER) LIMIT ?""",
            (ref.book, ref.chapter, ref.verse, ref.end or ref.verse, limit),
        ).fetchall()
    return [r["id"] for r in rows]
