"""Regression harness for retrieval quality.

The point is to make chunking, fusion and reranking tunable against a number
instead of vibes. Each case is a query plus the citation(s) that should come
back; the report gives hit@1, recall@n and MRR.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import ROOT
from . import index as idx
from . import search as search_mod

DEFAULT_QUERIES = ROOT / "tests" / "queries.yaml"


def load_cases(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else DEFAULT_QUERIES
    if not path.exists():
        raise SystemExit(f"No query set at {path}")
    cases = yaml.safe_load(path.read_text()) or []
    return [c for c in cases if c.get("query") and c.get("expect")]


def matches(result, expected: list[str]) -> bool:
    haystack = f"{result.citation} {result.title}".lower()
    return any(exp.lower() in haystack for exp in expected)


def run_eval(
    path: Path | None = None,
    *,
    n: int = 10,
    use_hyde: bool = True,
    use_rerank: bool = True,
) -> dict:
    cases = load_cases(path)
    conn = idx.connect(readonly=True)
    vectors = idx.load_vectors()

    hits_at_1 = 0
    recall = 0
    reciprocal = 0.0
    rows = []

    for case in cases:
        expected = case["expect"]
        if isinstance(expected, str):
            expected = [expected]

        results = search_mod.search(
            case["query"],
            n=n,
            filters=search_mod.Filters(source=case.get("source")),
            use_hyde=use_hyde,
            use_rerank=use_rerank,
            conn=conn,
            vectors=vectors,
        )

        rank = next(
            (i for i, r in enumerate(results, start=1) if matches(r, expected)), None
        )
        if rank == 1:
            hits_at_1 += 1
        if rank:
            recall += 1
            reciprocal += 1 / rank

        rows.append((case["query"], expected[0], rank))
        marker = "✓" if rank == 1 else (f"{rank}" if rank else "✗")
        print(f"  {marker:>3}  {case['query'][:64]:<66} -> {expected[0]}")

    total = len(cases) or 1
    report = {
        "cases": len(cases),
        "hit@1": hits_at_1 / total,
        f"recall@{n}": recall / total,
        "mrr": reciprocal / total,
    }

    print()
    print(f"  cases     {report['cases']}")
    print(f"  hit@1     {report['hit@1']:.0%}")
    print(f"  recall@{n} {report[f'recall@{n}']:.0%}")
    print(f"  MRR       {report['mrr']:.3f}")
    misses = [r for r in rows if r[2] is None]
    if misses:
        print(f"\n  missed entirely ({len(misses)}):")
        for query, expected, _ in misses:
            print(f"    {query}  ->  {expected}")

    return report
