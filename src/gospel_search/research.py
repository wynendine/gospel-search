"""Execute a Plan with the retrieval strategy its intent calls for."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import COMPARE_BUDGET, ENUMERATE_CAP, MAX_PER_DOC, THEMATIC_BUDGET
from . import index as idx
from . import reference as ref
from .plan import Plan
from .search import (
    Coverage,
    Filters,
    Result,
    diversify,
    hydrate,
    rerank,
    search_with_coverage,
)


@dataclass
class Findings:
    plan: Plan
    results: list[Result]
    coverage: Coverage
    exhaustive: bool = False  # every match is present, not just the best ones
    total_matches: int | None = None  # only meaningful when exhaustive
    groups: dict[str, list[Result]] | None = None  # compare: results per entity
    breakdown: list[tuple[str, int]] | None = None  # enumerate: where matches fall
    breakdown_label: str = ""
    note: str = ""  # a caveat the answer must not paper over


def resolve_reference(conn, query: str) -> Findings | None:
    """Return the exact verses if the query is a scripture reference.

    A reference has one right answer, so it never reaches the planner or an
    embedding — no LLM call, no ranking, no cost. Semantic search was actively
    wrong here: "D&C 121:7" came back as 121:40 and "1 Ne 3:7" as 2 Nephi 7:10.
    """
    parsed = ref.parse(query, conn)
    if parsed is None:
        return None
    ids = ref.lookup(conn, parsed)
    if not ids:
        return None

    results = hydrate(conn, [(i, 0.0, {}) for i in ids])
    plan = Plan(intent="reference", topic=parsed.citation())
    return Findings(
        plan=plan,
        results=results,
        coverage=Coverage(
            passages=len(results),
            documents=len({r.doc_id for r in results}),
            candidates=len(ids),
        ),
        exhaustive=True,
        total_matches=len(ids),
    )


def cited_by(conn, plan: Plan, limit: int = 40) -> Findings | None:
    """Talks that cite a scripture — a graph lookup, not a search.

    Reads the citation graph built from talk footnotes and inline references,
    so the answer is the actual set of talks that quote the verse rather than
    talks that happen to discuss the same subject.
    """
    from . import citations as cite

    parsed = ref.parse(plan.topic, conn)
    if parsed is None:
        return None

    rows = cite.citing_talks(conn, parsed, plan.filters, limit=limit)
    ids = [r["id"] for r in rows]
    if not ids:
        return Findings(
            plan=plan, results=[], coverage=Coverage(),
            note=f"No talk in the corpus cites {parsed.citation()}.",
        )

    # Show the paragraph that does the citing, not the talk's opening — the
    # answer to "which talks cite Alma 32:21" is the sentences that quote it.
    chunk_ids, seen = [], set()
    for row in rows:
        cid = row["src_chunk_id"]
        if cid and cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)
    results = hydrate(conn, [(i, 0.0, {}) for i in chunk_ids])
    order = {cid: i for i, cid in enumerate(chunk_ids)}
    results.sort(key=lambda r: order.get(r.chunk_id, 999))

    return Findings(
        plan=plan,
        results=results,
        coverage=Coverage(
            passages=len(results), documents=len(results), candidates=len(rows)
        ),
        exhaustive=len(rows) < limit,
        total_matches=len(rows),
    )


def enumerate_matches(conn, plan: Plan, limit: int = ENUMERATE_CAP) -> Findings:
    """Every chunk matching the literal terms, in canonical order.

    This is the one query shape where ranking is the wrong tool: "every
    reference to charity" wants completeness and scripture order, not the ten
    best. FTS5 can answer it exactly, so it does — and the count is real.

    Two honesty problems this has to handle. Matching is stemmed, so "faith"
    also catches "faithful" and the count is larger than a literal one; that is
    usually wanted but must be said. And plenty of terms are far too common to
    list — "faith" matches 16,081 chunks — so past the cap the answer becomes a
    distribution plus examples rather than a truncated list pretending to be
    an index.
    """
    where, params = plan.filters.sql()
    joiner = " AND " if plan.match_mode == "all" else " OR "
    match = joiner.join(f'"{t}"' for t in plan.literal_terms)

    total = conn.execute(
        f"""SELECT COUNT(*) FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? AND {where}""",
        [match, *params],
    ).fetchone()[0]

    stemmed = (
        "Matching is stemmed, so a term also catches its word forms "
        '("faith" includes "faithful").'
    )

    if total <= limit:
        rows = conn.execute(
            f"""SELECT c.id FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE chunks_fts MATCH ? AND {where}
                ORDER BY c.doc_id, c.ordinal""",
            [match, *params],
        ).fetchall()
        results = hydrate(conn, [(r["id"], 0.0, {}) for r in rows])
        return Findings(
            plan=plan,
            results=results,
            coverage=Coverage(
                passages=len(results),
                documents=len({r.doc_id for r in results}),
                candidates=total,
            ),
            exhaustive=True,
            total_matches=total,
            note=stemmed,
        )

    # Too many to list. Report where they fall, and show the best examples
    # rather than the first N in canonical order, which would all come from
    # Genesis and tell you nothing.
    scripture = plan.filters.source == "scriptures"
    group_by, label = (
        ("d.book", "matches by book") if scripture else ("SUBSTR(d.date,1,3)", "matches by decade")
    )
    breakdown = [
        (f"{r[0]}0s" if not scripture and r[0] else str(r[0]), r[1])
        for r in conn.execute(
            f"""SELECT {group_by} AS grp, COUNT(*) AS n FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE chunks_fts MATCH ? AND {where}
                GROUP BY grp ORDER BY n DESC LIMIT 25""",
            [match, *params],
        )
    ]

    rows = conn.execute(
        f"""SELECT f.rowid AS id FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? AND {where}
            ORDER BY bm25(chunks_fts, 4.0, 2.0, 1.0) LIMIT 20""",
        [match, *params],
    ).fetchall()
    results = hydrate(conn, [(r["id"], 0.0, {}) for r in rows])

    return Findings(
        plan=plan,
        results=results,
        coverage=Coverage(
            passages=len(results),
            documents=len({r.doc_id for r in results}),
            candidates=total,
        ),
        exhaustive=False,
        total_matches=total,
        breakdown=breakdown,
        breakdown_label=label,
        note=(
            f"{total:,} passages match — far too many to list, so this is where "
            f"they fall plus the {len(results)} strongest examples, NOT a "
            f"complete index. {stemmed} Narrow the scope (a book, a volume, a "
            "speaker, a date range) to get a complete list."
        ),
    )


def compare_entities(conn, vectors, plan: Plan, per_entity: int) -> Findings:
    """One retrieval per entity, so neither side is crowded out.

    A single blended search for "compare Bednar and Holland on faith" returns
    passages about faith, most of them by neither man. Scoping per speaker is
    what turns the answer into a comparison of two people rather than of
    whoever ranked well.
    """
    groups: dict[str, list[Result]] = {}
    merged: list[Result] = []
    considered = 0

    for entity in plan.entities:
        scoped = replace(plan.filters, speaker=entity)
        found, cover = search_with_coverage(
            plan.topic or entity,
            n=per_entity,
            filters=scoped,
            max_per_doc=1,  # breadth across an entity's talks, not depth in one
            conn=conn,
            vectors=vectors,
        )
        groups[entity] = found
        merged.extend(found)
        considered += cover.candidates

    missing = [e for e in plan.entities if not groups.get(e)]
    return Findings(
        plan=plan,
        results=merged,
        coverage=Coverage(
            passages=len(merged),
            documents=len({r.doc_id for r in merged}),
            candidates=considered,
        ),
        groups=groups,
        note=(
            f"Nothing found for: {', '.join(missing)}."
            if missing
            else ""
        ),
    )


def thematic_survey(
    conn, vectors, plan: Plan, query: str, n: int, per_facet: int, use_rerank: bool
) -> Findings:
    """Retrieve per facet, then rank the union against the original question.

    One broad query embeds to the centre of a subject and returns that centre.
    Measured on "the gathering of Israel": a single query found 12 sources;
    the same question split into facets surfaced 39, so the single query was
    seeing 31% of the material.

    Facets are retrieved without HyDE or reranking — they are already specific,
    and five extra Claude calls to order six passages each is not worth it.
    One rerank pass over the union puts the best material on top while the
    facets guarantee the breadth got in.
    """
    pool: dict[int, Result] = {}
    considered = 0

    for facet in [plan.topic, *plan.facets]:
        found, cover = search_with_coverage(
            facet,
            n=per_facet,
            filters=plan.filters,
            use_hyde=False,
            use_rerank=False,
            max_per_doc=1,
            conn=conn,
            vectors=vectors,
        )
        considered += cover.candidates
        for r in found:
            # First facet to surface a passage gets credit for it.
            if r.chunk_id not in pool:
                r.facet = facet
                pool[r.chunk_id] = r

    results = list(pool.values())
    if use_rerank and results:
        results = rerank(query, results)
    results = diversify(results, 1, n)  # one passage per source: breadth is the point

    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.facet or plan.topic, []).append(r)

    return Findings(
        plan=plan,
        results=results,
        coverage=Coverage(
            passages=len(results),
            documents=len({r.doc_id for r in results}),
            candidates=considered,
        ),
        groups=groups,
        note=(
            "This is what the retrieved passages show across "
            f"{len(groups)} angles, not the whole of what has been taught."
        ),
    )


def investigate(
    query: str,
    plan: Plan,
    *,
    n: int = 10,
    use_hyde: bool = True,
    use_rerank: bool = True,
    conn=None,
    vectors=None,
) -> Findings:
    """Run the retrieval strategy that the plan's intent calls for."""
    conn = conn or idx.connect(readonly=True)
    vectors = idx.load_vectors() if vectors is None else vectors

    if plan.intent == "reference":
        found = resolve_reference(conn, plan.topic)
        if found is not None:
            return found

    if plan.intent == "citations":
        found = cited_by(conn, plan)
        if found is not None:
            return found

    if plan.intent == "enumerate" and plan.literal_terms:
        return enumerate_matches(conn, plan)

    if plan.intent == "compare" and len(plan.entities) >= 2:
        return compare_entities(conn, vectors, plan, COMPARE_BUDGET)

    if plan.intent == "thematic" and plan.facets:
        return thematic_survey(
            conn, vectors, plan, query, max(n, 16), THEMATIC_BUDGET, use_rerank
        )

    # lookup wants the best passages, which often means several from the one
    # right talk; a thematic question with no facets falls back to breadth.
    max_per_doc = 1 if plan.intent == "thematic" else MAX_PER_DOC
    if plan.intent == "thematic":
        n = max(n, 12)

    results, cover = search_with_coverage(
        plan.topic or query,
        n=n,
        filters=plan.filters,
        use_hyde=use_hyde,
        use_rerank=use_rerank,
        max_per_doc=max_per_doc,
        conn=conn,
        vectors=vectors,
    )

    note = ""
    if plan.intent == "enumerate" and not plan.literal_terms:
        note = (
            "This asks for a concept rather than a fixed word, so the results "
            "are the best matches, not a complete list."
        )

    return Findings(plan=plan, results=results, coverage=cover, note=note)
