"""Build the talk -> scripture citation graph from already-cached pages.

Re-parses what the crawl already downloaded: no refetching, no re-embedding.

Each link records the *citing paragraph*, not just the citing talk. Without
that, a reverse lookup can only show a talk's opening pleasantries — the answer
to "which talks cite Alma 32:21" should be the sentences that quote it.
Footnotes carry a `pid` (the paragraph's data-aid), and the talk body maps
data-aid to the anchor the chunk is stored under.
"""

from __future__ import annotations

import sys

from bs4 import BeautifulSoup

from . import citations as cite
from .reference import aliases
from .sources import church_api


def _paragraph_anchors(payload: dict) -> dict[str, str]:
    """data-aid -> anchor id, so a footnote's pid finds its paragraph."""
    body = payload.get("content", {}).get("body", "")
    soup = BeautifulSoup(body, "lxml")
    return {
        node["data-aid"]: node["id"]
        for node in soup.select("p[data-aid][id]")
    }


def build(conn, *, include_prose: bool = True) -> tuple[int, int]:
    """Populate the citations table. Returns (links, talks with links)."""
    chapters = {
        (r["book"], r["chapter"]): r["id"]
        for r in conn.execute(
            "SELECT id, book, chapter FROM documents WHERE kind='chapter'"
        )
    }
    verses = {
        (r["book"], r["chapter"], int(r["anchor"])): r["id"]
        for r in conn.execute(
            """SELECT c.id, d.book, d.chapter, c.anchor
                 FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.kind = 'verse'"""
        )
    }
    alias_table = aliases(conn)

    talks = conn.execute(
        "SELECT id, key FROM documents WHERE kind='talk' ORDER BY date"
    ).fetchall()

    conn.execute("DELETE FROM citations")
    rows: list[tuple] = []
    with_links = 0

    for i, talk in enumerate(talks, start=1):
        payload = church_api.fetch(talk["key"])
        if not payload:
            continue

        # anchor -> chunk id for this talk's paragraphs
        chunk_by_anchor = {
            r["anchor"]: r["id"]
            for r in conn.execute(
                "SELECT id, anchor FROM chunks WHERE doc_id = ?", (talk["id"],)
            )
        }

        found: list[tuple[cite.Citation, int | None]] = []

        anchors = _paragraph_anchors(payload)
        for note in (payload.get("content", {}).get("footnotes") or {}).values():
            src_chunk = chunk_by_anchor.get(anchors.get(str(note.get("pid") or ""), ""))
            for c in cite.from_footnotes({"content": {"footnotes": {"n": note}}}):
                found.append((c, src_chunk))

        if include_prose and not found:
            # Only when a talk has no footnotes at all — otherwise the same
            # reference lands in the graph twice.
            for r in conn.execute(
                "SELECT id, display_text FROM chunks WHERE doc_id = ? ORDER BY ordinal",
                (talk["id"],),
            ):
                for c in cite.from_prose(r["display_text"], alias_table):
                    found.append((c, r["id"]))

        seen = set()
        for c, src_chunk in found:
            tgt_doc = chapters.get((c.book, c.chapter))
            targets = [verses.get((c.book, c.chapter, v)) for v in c.verses] or [None]
            for tgt_chunk in targets:
                key = (talk["id"], src_chunk, tgt_doc, tgt_chunk, c.text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((*key, c.origin))
        if seen:
            with_links += 1

        if i % 250 == 0:
            sys.stdout.write(f"\r  {i}/{len(talks)} talks, {len(rows):,} links")
            sys.stdout.flush()

    conn.executemany(
        """INSERT OR IGNORE INTO citations
             (src_doc_id, src_chunk_id, tgt_doc_id, tgt_chunk_id, citation, origin)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    print(f"\r  {len(talks)} talks, {len(rows):,} links from {with_links} talks")
    return len(rows), with_links
