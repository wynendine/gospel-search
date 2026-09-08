"""Unit tests for the pure logic — no network, no API keys, no index.

Run: .venv/bin/python tests/test_units.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from gospel_search import chunk as chunker
from gospel_search import index as idx
from gospel_search.search import Filters, fts_query, fuse
from gospel_search.sources import conference

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# --- FTS5 query sanitization ----------------------------------------------
# The reason this exists: an apostrophe or a stray quote in a natural query
# would otherwise be parsed as an FTS operator and raise mid-search.

def test_fts_query() -> None:
    check("fts: plain words", fts_query("natural man") == '"natural" OR "man"')
    check("fts: apostrophes survive", fts_query("Lehi's dream") == '"Lehi\'s" OR "dream"')
    check(
        "fts: punctuation dropped",
        fts_query('what did he say -- "faith"?') == '"what" OR "did" OR "he" OR "say" OR "faith"',
    )
    check("fts: single chars dropped", fts_query("a b faith") == '"faith"')
    check("fts: empty query is empty", fts_query("?! -- ,") == "")
    check("fts: token cap respected", len(fts_query(" ".join(f"word{i}" for i in range(50))).split(" OR ")) == 24)


# --- Reciprocal rank fusion ------------------------------------------------

def test_fuse() -> None:
    dense = [(10, 0.9), (20, 0.8), (30, 0.7)]
    lexical = [(30, 5.0), (40, 4.0)]
    fused = fuse(dense, lexical, k=60)
    order = [cid for cid, _, _ in fused]

    check("rrf: appearing in both halves wins", order[0] == 30, f"got {order}")
    check("rrf: every candidate survives", set(order) == {10, 20, 30, 40})

    ranks = {cid: r for cid, _, r in fused}
    check("rrf: records both source ranks", ranks[30] == {"dense": 3, "lexical": 1})
    check("rrf: records a single source rank", ranks[10] == {"dense": 1})
    check("rrf: empty input is empty", fuse([], []) == [])


# --- Filters ---------------------------------------------------------------

def test_filters() -> None:
    where, params = Filters().sql()
    check("filters: empty is a no-op", where == "1=1" and params == [])
    check("filters: empty is inactive", Filters().active is False)

    where, params = Filters(speaker="Holland", after=2010).sql()
    check("filters: composes with AND", " AND " in where)
    check("filters: speaker is a LIKE", params[0] == "%Holland%")
    check("filters: year becomes a date bound", params[1] == "2010-01-01")

    where, _ = Filters(source="scriptures").sql()
    check("filters: scriptures covers verses and summaries", "'verse'" in where and "'summary'" in where)


# --- Chunk windowing -------------------------------------------------------

def test_windowing() -> None:
    talk = SimpleNamespace(
        uri="/general-conference/2020/04/test",
        title="A Title",
        speaker="President Test Speaker",
        role="A Role",
        year=2020,
        month="04",
        date="2020-04-01",
        conference="April 2020",
        url="https://example.com/talk",
        paragraphs=[("p1", "First."), ("p2", "Second."), ("p3", "Third."), ("p4", "Fourth.")],
    )
    document, chunks = chunker.chunk_talk(talk)

    check("chunk: one chunk per paragraph", len(chunks) == 4)
    check("chunk: display is the center only", chunks[1].display_text == "Second.")
    check(
        "chunk: window spans neighbours",
        chunks[1].window_text == "First.\n\nSecond.\n\nThird.",
    )
    check(
        "chunk: first window has no left neighbour",
        chunks[0].window_text == "First.\n\nSecond.",
    )
    check(
        "chunk: last window has no right neighbour",
        chunks[3].window_text == "Third.\n\nFourth.",
    )
    check(
        "chunk: provenance is inside the embedded text",
        "President Test Speaker" in chunks[0].embed_text
        and "April 2020" in chunks[0].embed_text,
    )
    check("chunk: url carries the paragraph anchor", chunks[2].url.endswith("#p3"))
    check("chunk: document metadata survives", document.speaker == "President Test Speaker")


# --- Byline cleanup --------------------------------------------------------

def test_byline() -> None:
    strip = conference._BYLINE_LEAD.sub
    check("byline: 'By' stripped", strip("", "By President Nelson") == "President Nelson")
    check(
        "byline: 'Presented by' stripped",
        strip("", "Presented by President Oaks") == "President Oaks",
    )
    check(
        "byline: a name starting with 'By' is untouched",
        strip("", "Byron K. Smith") == "Byron K. Smith",
    )


# --- Conference periods ----------------------------------------------------

def test_periods() -> None:
    import datetime as dt

    periods = conference.conference_periods(through=dt.date(2025, 6, 1))
    check("periods: starts at April 1971", periods[0] == (1971, "04"))
    check("periods: stops before an unheld conference", periods[-1] == (2025, "04"))
    check("periods: two per full year", periods.count((2000, "04")) == 1 and (2000, "10") in periods)

    through_october = conference.conference_periods(through=dt.date(2025, 10, 15))
    check("periods: includes October once it arrives", through_october[-1] == (2025, "10"))


# --- Index schema ----------------------------------------------------------

def test_schema() -> None:
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    idx.init(conn)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("schema: tables created", {"documents", "chunks"} <= tables)

    document = chunker.Document(key="/k", kind="talk", title="T", url="u", speaker="S")
    chunks = [
        chunker.Chunk("/k", "talk", "p1", 0, "cite", "u#p1", "text one", "win", "embed one"),
    ]
    inserted = idx.add_document(conn, document, chunks)
    check("schema: insert reports chunk count", inserted == 1)

    again = idx.add_document(conn, document, chunks)
    check("schema: re-adding an unchanged document is a no-op", again == 0)
    check("schema: no duplicate rows", conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1)

    changed = [
        chunker.Chunk("/k", "talk", "p1", 0, "cite", "u#p1", "text one", "win", "embed CHANGED"),
    ]
    check("schema: a changed document is replaced", idx.add_document(conn, document, changed) == 1)
    check("schema: still one row after replace", conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1)

    idx.rebuild_fts(conn)
    hit = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ('"text"',)
    ).fetchone()
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()[0]
    check("schema: fts rowid equals the chunk id", hit is not None and hit[0] == chunk_id)


# --- Incremental chunk updates ---------------------------------------------
# The property that matters: a chunk whose embed_text is unchanged keeps its
# row id, and therefore its (expensive) vector.

def test_incremental() -> None:
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    idx.init(conn)

    doc = chunker.Document(key="/ch", kind="chapter", title="Ch 1", url="u")
    def c(anchor, embed):
        return chunker.Chunk("/ch", "verse", anchor, 0, f"cite {anchor}", "u", "d", "w", embed)

    idx.add_document(conn, doc, [c("1", "one"), c("2", "two")])
    ids = {r["anchor"]: r["id"] for r in conn.execute("SELECT anchor, id FROM chunks")}
    check("incremental: initial insert", len(ids) == 2)

    # Add a third chunk; the first two must keep their ids (and their vectors).
    added = idx.add_document(conn, doc, [c("1", "one"), c("2", "two"), c("summary", "sum")])
    after = {r["anchor"]: r["id"] for r in conn.execute("SELECT anchor, id FROM chunks")}
    check("incremental: only the new chunk is inserted", added == 1, f"got {added}")
    check("incremental: siblings keep their ids", after["1"] == ids["1"] and after["2"] == ids["2"])
    check("incremental: new chunk exists", "summary" in after)

    # Change one chunk's text; only that one is re-inserted.
    added = idx.add_document(conn, doc, [c("1", "ONE CHANGED"), c("2", "two"), c("summary", "sum")])
    final = {r["anchor"]: r["id"] for r in conn.execute("SELECT anchor, id FROM chunks")}
    check("incremental: only the changed chunk re-inserts", added == 1, f"got {added}")
    check("incremental: changed chunk gets a new id", final["1"] != after["1"])
    check("incremental: unchanged chunk keeps its id", final["2"] == after["2"])

    # Drop a chunk; it disappears without disturbing the rest.
    idx.add_document(conn, doc, [c("2", "two")])
    remaining = [r["anchor"] for r in conn.execute("SELECT anchor FROM chunks")]
    check("incremental: removed chunks are deleted", remaining == ["2"], f"got {remaining}")

    # Document metadata is refreshed in place, not duplicated.
    doc2 = chunker.Document(key="/ch", kind="chapter", title="Renamed", url="u2")
    idx.add_document(conn, doc2, [c("2", "two")])
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    title = conn.execute("SELECT title FROM documents").fetchone()[0]
    check("incremental: document not duplicated", docs == 1)
    check("incremental: document metadata updated", title == "Renamed")


# --- Diversity cap ---------------------------------------------------------
# The property: relevance still sets the order; the cap only decides who gets
# crowded out when one document would otherwise take every slot.

def test_diversify() -> None:
    from gospel_search.search import diversify

    def r(chunk_id, doc_id):
        return chunker.Chunk("k", "talk", str(chunk_id), 0, "c", "u", "d", "w", "e"), doc_id

    class Fake:
        def __init__(self, cid, did): self.chunk_id, self.doc_id = cid, did
        def __repr__(self): return f"c{self.chunk_id}/d{self.doc_id}"

    # One document dominating the ranking.
    ranked = [Fake(1,10), Fake(2,10), Fake(3,10), Fake(4,10), Fake(5,20), Fake(6,30)]
    out = diversify(ranked, max_per_doc=2, n=4)
    check("diversify: caps one document", [x.doc_id for x in out] == [10,10,20,30], f"got {out}")
    check("diversify: preserves rank order", [x.chunk_id for x in out][:2] == [1,2])

    out = diversify(ranked, max_per_doc=0, n=3)
    check("diversify: 0 disables the cap", [x.chunk_id for x in out] == [1,2,3])

    # Too few distinct documents to fill n — backfill rather than return short.
    thin = [Fake(1,10), Fake(2,10), Fake(3,10)]
    out = diversify(thin, max_per_doc=1, n=3)
    check("diversify: backfills when sources run out", len(out) == 3, f"got {out}")
    check("diversify: backfill keeps the best first", out[0].chunk_id == 1)

    check("diversify: empty input", diversify([], 2, 5) == [])


# --- Plan describe / fallback ----------------------------------------------

def test_plan() -> None:
    from gospel_search.plan import Plan
    from gospel_search.search import Filters

    p = Plan(intent="enumerate", topic="charity", literal_terms=["charity"],
             filters=Filters(volume="Book of Mormon", source="scriptures"))
    d = p.describe()
    check("plan: describe names the intent", "intent=enumerate" in d)
    check("plan: describe shows scoping", "volume=Book of Mormon" in d)
    check("plan: describe omits unset filters", "speaker=" not in d)

    check("plan: default intent is lookup", Plan().intent == "lookup")
    check("plan: default filters are inert", Plan().filters.active is False)


# --- Scripture reference parsing -------------------------------------------
# A reference has one right answer, so this must never fall through to search.

def test_reference() -> None:
    from gospel_search.reference import PATTERN, _normalize

    def parts(text):
        m = PATTERN.match(text)
        if not m: return None
        return (m.group("book").strip(), m.group("chapter"), m.group("verse"), m.group("end"))

    check("ref: book chapter verse", parts("Alma 32:21") == ("Alma","32","21",None))
    check("ref: ampersand book", parts("D&C 121:7") == ("D&C","121","7",None))
    check("ref: numbered book", parts("1 Ne 3:7") == ("1 Ne","3","7",None))
    check("ref: trailing period", parts("1 Ne. 3:7") == ("1 Ne.","3","7",None))
    check("ref: verse range", parts("Moroni 10:3-5") == ("Moroni","10","3","5"))
    check("ref: en dash range", parts("Moroni 10:3\u20135") == ("Moroni","10","3","5"))
    check("ref: chapter only", parts("Alma 32") == ("Alma","32",None,None))
    check("ref: hyphenated book", parts("JS-H 1:17") == ("JS-H","1","17",None))
    check("ref: case insensitive", parts("moroni 10:4") == ("moroni","10","4",None))

    check("ref: prose is not a reference", parts("the natural man is an enemy to God") is None)
    check("ref: question is not a reference", parts("what did Nelson say about faith") is None)
    check("ref: bare number is not a reference", parts("121") is None)

    check("ref: normalize folds punctuation", _normalize("1 Ne.") == "1ne")
    check("ref: normalize keeps ampersand", _normalize("D&C") == "d&c")
    check("ref: normalize folds case", _normalize("Alma") == _normalize("ALMA"))


# --- Citation extraction ---------------------------------------------------

def test_citations() -> None:
    from gospel_search import citations as cite

    # Verse ids out of the href query string.
    check("cite: single verse", cite._verses("?lang=eng&id=p4#p4") == (4,))
    check("cite: verse range", cite._verses("?lang=eng&id=p8-p9#p8") == (8, 9))
    check("cite: url-encoded comma list", cite._verses("?id=p12%2Cp14#p12") == (12, 14))
    check("cite: chapter-level has no verses", cite._verses("?lang=eng") == ())
    check("cite: absurd range collapses", cite._verses("?id=p1-p9999") == (1,))

    # Inline prose extraction is gated on the book resolving.
    # Keys are _normalize()d, which preserves "&" — the real table carries
    # both "d&c" and "dc" for exactly this reason.
    aliases = {"d&c": "Doctrine and Covenants", "dc": "Doctrine and Covenants",
               "alma": "Alma"}
    found = cite.from_prose("as it says in Alma 32:21, and also D&C 64:9", aliases)
    books = sorted((c.book, c.chapter, c.verses) for c in found)
    check("cite: finds inline references", len(found) == 2, f"got {found}")
    check("cite: resolves abbreviations", ("Doctrine and Covenants", 64, (9,)) in books)

    noise = cite.from_prose("the meeting ran 10:30 to 11:45 on April 6", aliases)
    check("cite: unknown book is not a citation", noise == [], f"got {noise}")

    ranged = cite.from_prose("see Alma 32:21-23", aliases)
    check("cite: inline range expands", ranged and ranged[0].verses == (21, 22, 23))


def main() -> int:
    for test in (
        test_fts_query,
        test_fuse,
        test_filters,
        test_windowing,
        test_byline,
        test_periods,
        test_schema,
        test_incremental,
        test_diversify,
        test_plan,
        test_reference,
        test_citations,
    ):
        print(f"\n{test.__name__}")
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
