"""Execute a Plan with the retrieval strategy its intent calls for."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import COMPARE_BUDGET, ENUMERATE_CAP, MAX_PER_DOC
from . import index as idx
from . import reference as ref
from .plan import Plan
from .search import (
    Coverage,
    Filters,
    Result,
    hydrate,
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


def enumerate_matches(conn, plan: Plan, limit: int = ENUMERATE_CAP) -> Findings:
    """Every chunk containing the literal term, in canonical order.

    This is the one query shape where ranking is the wrong tool: "every
    reference to charity" wants completeness and scripture order, not the ten
    best. FTS5 can answer it exactly, so it does — and the count is real.
    """
    where, params = plan.filters.sql()
    match = " OR ".join(f'"{t}"' for t in plan.literal_terms)

    total = conn.execute(
        f"""SELECT COUNT(*) FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? AND {where}""",
        [match, *params],
    ).fetchone()[0]

    rows = conn.execute(
        f"""SELECT c.id FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? AND {where}
            ORDER BY c.doc_id, c.ordinal
            LIMIT ?""",
        [match, *params, limit],
    ).fetchall()

    results = hydrate(conn, [(r["id"], 0.0, {}) for r in rows])
    complete = total <= limit

    return Findings(
        plan=plan,
        results=results,
        coverage=Coverage(
            passages=len(results),
            documents=len({r.doc_id for r in results}),
            candidates=total,
        ),
        exhaustive=complete,
        total_matches=total,
        note=(
            ""
            if complete
            else f"Showing the first {limit} of {total} matches in canonical order."
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

    if plan.intent == "enumerate" and plan.literal_terms:
        return enumerate_matches(conn, plan)

    if plan.intent == "compare" and len(plan.entities) >= 2:
        return compare_entities(conn, vectors, plan, COMPARE_BUDGET)

    # thematic wants breadth across sources; lookup wants the best passages,
    # which often means several from the one right talk.
    if plan.intent == "thematic":
        n = max(n, 12)
        max_per_doc = 1
    else:
        max_per_doc = MAX_PER_DOC

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
