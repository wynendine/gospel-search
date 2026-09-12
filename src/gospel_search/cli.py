"""Command line entry point: build / search / eval / serve / stats."""

from __future__ import annotations

import argparse
import textwrap
import time
from dataclasses import dataclass

from . import answer as answer_mod
from . import cache as cache_mod
from . import usage as usage_mod
from . import build as build_mod
from . import index as idx
from . import research as research_mod
from . import search as search_mod
from .config import MAX_PER_DOC
from .plan import Plan, plan as make_plan

BOLD, DIM, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"


@dataclass
class _Row:
    """A cached result, shaped enough for the printer."""
    citation: str; url: str; speaker: str; date: str
    rerank: float | None; dense_rank: int | None; lexical_rank: int | None
    display_text: str


def wrap(text: str, width: int = 88, indent: str = "     ") -> str:
    return textwrap.fill(
        text, width=width, initial_indent=indent, subsequent_indent=indent
    )


def cmd_build(args) -> None:
    conn = idx.connect()
    idx.init(conn)

    if args.only in (None, "scriptures"):
        build_mod.build_scriptures(
            conn, verify=not args.no_verify
        )
    if args.only in (None, "talks"):
        periods = None
        if args.since:
            periods = [
                (y, m)
                for y, m in build_mod.conference.conference_periods()
                if y >= args.since
            ]
        build_mod.build_talks(conn, periods=periods)

    if args.citations:
        from . import build_citations

        print("Building the citation graph from cached pages...")
        build_citations.build(conn)

    build_mod.finalize(conn, skip_embed=args.no_embed)


def cmd_search(args) -> None:
    filters = search_mod.Filters(
        speaker=args.speaker,
        after=args.after,
        before=args.before,
        source=args.source,
        volume=args.volume,
        book=args.book,
    )
    # A bare scripture reference has exactly one right answer, so it resolves
    # directly — no planner, no embedding, no synthesis.
    conn = idx.connect(readonly=True)
    findings = research_mod.resolve_reference(conn, args.query)
    if findings is not None:
        print(f"{DIM}     [reference] {findings.coverage.summary()}{RESET}")
        _print_results(findings.results)
        return

    usage_mod.start_run()
    flags = {
        "n": args.n, "hyde": not args.no_hyde, "rerank": not args.no_rerank,
        "answer": not args.no_answer, "plan": not args.no_plan,
        "mpd": args.max_per_doc,
    }
    ckey = cache_mod.key(args.query, filters, flags)
    cached = None if args.fresh else cache_mod.get(ckey)
    if cached:
        age = time.time() - cached["_cached_at"]
        unit = f"{age/86400:.0f}d" if age > 86400 else f"{age/3600:.0f}h" if age > 3600 else f"{age/60:.0f}m"
        if cached["answer"]:
            print(f"\n{BOLD}{cached['answer']}{RESET}\n")
            print(DIM + "─" * 88 + RESET)
        print(f"{DIM}     [{cached['intent']}] {cached['coverage']}  ·  cached {unit} ago (free){RESET}")
        _print_results([_Row(**r) for r in cached["results"]])
        return

    if args.no_plan:
        plan = Plan(topic=args.query, filters=filters)
    else:
        plan = make_plan(args.query, override=filters)
        if args.explain:
            print(f"{DIM}     plan: {plan.describe()}{RESET}")

    try:
        findings = research_mod.investigate(
            args.query,
            plan,
        n=args.n,
            use_hyde=not args.no_hyde,
            use_rerank=not args.no_rerank,
        )
    except usage_mod.BudgetExceeded as limit:
        print(f"{YELLOW}Monthly limit reached.{RESET} {limit}")
        return
    results = findings.results

    if not results:
        print("No matches.")
        return

    text = ""
    if not args.no_answer:
        text = answer_mod.answer(args.query, findings, conn=conn)
        print(f"\n{BOLD}{text}{RESET}\n")
        print(DIM + "─" * 88 + RESET)

    # What the answer actually saw. Without this the output reads like a survey
    # of the whole corpus regardless of how thin the evidence was.
    cache_mod.put(ckey, args.query, {
        "answer": text if not args.no_answer else "",
        "intent": plan.intent,
        "coverage": findings.coverage.summary(),
        "results": [
            {
                "citation": r.citation, "url": r.url, "speaker": r.speaker,
                "date": r.date, "rerank": r.rerank, "dense_rank": r.dense_rank,
                "lexical_rank": r.lexical_rank, "display_text": r.display_text,
            } for r in results
        ],
    })

    line = findings.coverage.summary()
    if findings.exhaustive:
        line = f"complete: all {findings.total_matches} matches · " + line
    elif findings.total_matches:
        line = f"{findings.total_matches:,} matches (showing examples) · " + line
    print(f"{DIM}     [{plan.intent}] {line}{RESET}")

    if findings.breakdown:
        print(f"\n{BOLD}{findings.breakdown_label}{RESET}")
        widest = max(len(g) for g, _ in findings.breakdown)
        biggest = max(n for _, n in findings.breakdown)
        for group, count in findings.breakdown:
            bar = "\u2588" * max(1, round(28 * count / biggest))
            print(f"  {group:<{widest}}  {count:>6,}  {CYAN}{bar}{RESET}")

    _print_results(results)


def _print_results(results) -> None:
    for i, result in enumerate(results, start=1):
        meta = []
        if result.speaker:
            meta.append(result.speaker)
        if result.date:
            meta.append(result.date[:4])
        if result.rerank is not None:
            meta.append(f"rerank {result.rerank:.0f}/10")
        halves = "+".join(
            h
            for h, present in (
                ("dense", result.dense_rank),
                ("bm25", result.lexical_rank),
            )
            if present
        )
        meta.append(halves)

        print(f"\n{CYAN}[{i}] {result.citation}{RESET}")
        print(f"{DIM}     {' · '.join(meta)}{RESET}")
        print(wrap(result.display_text))
        print(f"{DIM}     {result.url}{RESET}")
    print()


def cmd_cites(args) -> None:
    from . import citations as cite
    from . import reference as ref

    conn = idx.connect(readonly=True)
    filters = search_mod.Filters(
        speaker=args.speaker, after=args.after, before=args.before
    )

    if args.most_cited:
        rows = cite.most_cited(conn, filters, limit=args.n)
        if not rows:
            print("No citations indexed — run `gospel build --citations`.")
            return
        print(f"{BOLD}Most-cited verses{RESET}")
        for r in rows:
            print(f"\n{CYAN}{r['talks']:>4} talks · {r['citation']}{RESET}")
            print(wrap(r["display_text"][:220]))
        return

    parsed = ref.parse(args.reference or "", conn)
    if parsed is None:
        print(f'Not a scripture reference: "{args.reference}"')
        return

    rows = cite.citing_talks(conn, parsed, filters, limit=args.n)
    print(f"{BOLD}{len(rows)} talks cite {parsed.citation()}{RESET}")

    passages = {
        r["id"]: r["display_text"]
        for r in conn.execute(
            "SELECT id, display_text FROM chunks WHERE id IN ("
            + ",".join("?" * len(rows)) + ")",
            [r["src_chunk_id"] for r in rows],
        )
    } if rows else {}

    for r in rows:
        print(f"\n{CYAN}{r['date'][:4]}  {r['title']}{RESET}")
        print(f"{DIM}     {r['speaker']} · cited as {r['citation']}{RESET}")
        text = passages.get(r["src_chunk_id"])
        if text:
            print(wrap(text[:280] + ("…" if len(text) > 280 else "")))
        print(f"{DIM}     {r['url']}{RESET}")
    print()


def cmd_spend(args) -> None:
    st = usage_mod.summary()
    print(f"{BOLD}Spend{RESET}")
    if st["limit"]:
        pct = st["spent"] / st["limit"]
        colour = YELLOW if pct > 0.8 else ""
        print(f"  this month : {colour}${st['spent']:.2f} of ${st['limit']:.2f} ({pct:.0%}){RESET}")
    else:
        print(f"  this month : ${st['spent']:.2f}  (no limit set)")
    print(f"  searches   : {st['searches']}")
    if st["searches"]:
        print(f"  per search : ${st['per_search']:.4f} average")

    if st["by_stage"]:
        print(f"\n{BOLD}By stage, this month{RESET}")
        for r in st["by_stage"]:
            print(f"  {r['stage']:<8}{r['calls']:>5} calls  ${r['cost']:>8.4f}")
    if len(st["by_month"]) > 1:
        print(f"\n{BOLD}By month{RESET}")
        for r in st["by_month"]:
            print(f"  {r['month']}  {r['calls']:>5} calls  ${r['cost']:>8.2f}")
    if not st["limit"]:
        print(f"\n{DIM}  Set a ceiling with:  export GOSPEL_MONTHLY_LIMIT=10{RESET}")


def cmd_cache(args) -> None:
    if args.clear:
        n = cache_mod.clear()
        print(f"Cleared {n:,} cached answers.")
        return
    if args.prune:
        n = cache_mod.clear(stale_only=True)
        print(f"Removed {n:,} answers from superseded index builds.")
        return
    st = cache_mod.stats()
    print(f"{BOLD}Query cache{RESET}")
    print(f"  entries : {st['entries']:,}")
    print(f"  size    : {st['bytes']/1e6:.1f} MB")
    if st["stale"]:
        print(f"  {YELLOW}stale (older index): {st['stale']:,} — `gospel cache --prune`{RESET}")
    print(f"\n  Each hit saves a full search (~$0.11 and ~20s).")


def cmd_stats(args) -> None:
    meta = idx.read_meta()
    if not meta:
        print("No index yet — run `gospel build`.")
        return
    print(f"{BOLD}Index{RESET}")
    print(f"  documents : {meta.get('documents', 0):,}")
    print(f"  chunks    : {meta.get('chunks', 0):,}")
    for kind, count in sorted(meta.get("by_kind", {}).items()):
        print(f"    {kind:<8}: {count:,}")
    print(f"  model     : {meta.get('embed_model')} @ {meta.get('dims')} dims")

    conn = idx.connect(readonly=True)
    pending = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded = 0"
    ).fetchone()[0]
    if pending:
        print(f"  {YELLOW}pending embeddings: {pending:,}{RESET}")


def cmd_compact(args) -> None:
    conn = idx.connect()
    before, after = idx.compact(conn)
    if before == after:
        print(f"Already compact ({after:,} rows).")
    else:
        print(f"Reclaimed {before - after:,} stranded rows ({before:,} -> {after:,}).")
    idx.assign_vector_rows(conn)
    idx.write_meta(conn, after)


def cmd_eval(args) -> None:
    from .evaluate import run_eval

    run_eval(path=args.file, n=args.n, use_hyde=not args.no_hyde,
             use_rerank=not args.no_rerank)


def cmd_serve(args) -> None:
    import uvicorn

    from .web import app

    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="gospel",
        description="Semantic search over General Conference talks and the scriptures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="ingest, embed, and index the corpus")
    build.add_argument("--only", choices=["talks", "scriptures"])
    build.add_argument("--since", type=int, help="only conferences from this year on")
    build.add_argument("--no-embed", action="store_true", help="ingest only")
    build.add_argument("--no-verify", action="store_true", help="skip the URL check")
    build.add_argument(
        "--citations", action="store_true",
        help="rebuild the talk->scripture citation graph from cached pages",
    )
    build.set_defaults(func=cmd_build)

    search = sub.add_parser("search", help="search the corpus")
    search.add_argument("query")
    search.add_argument("-n", type=int, default=10)
    search.add_argument("--speaker")
    search.add_argument("--after", type=int, metavar="YEAR")
    search.add_argument("--before", type=int, metavar="YEAR")
    search.add_argument("--source", choices=["talks", "scriptures"])
    search.add_argument("--volume", help='e.g. "Book of Mormon", "New Testament"')
    search.add_argument("--book", help='e.g. "Alma", "Moroni", "Isaiah"')
    search.add_argument(
        "--max-per-doc", type=int, default=None, dest="max_per_doc",
        help="cap passages from one talk/chapter (default 3; 0 disables)",
    )
    search.add_argument("--no-answer", action="store_true")
    search.add_argument(
        "--no-plan", action="store_true",
        help="skip query planning; treat the query as a plain lookup",
    )
    search.add_argument(
        "--explain", action="store_true", help="print the plan before searching"
    )
    search.add_argument("--no-hyde", action="store_true")
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument(
        "--fresh", action="store_true", help="ignore the cache and re-run the query"
    )
    search.set_defaults(func=cmd_search)

    cites = sub.add_parser("cites", help="which talks cite a scripture")
    cites.add_argument("reference", nargs="?", help='e.g. "Alma 32:21", "D&C 121"')
    cites.add_argument("-n", type=int, default=25)
    cites.add_argument("--speaker")
    cites.add_argument("--after", type=int, metavar="YEAR")
    cites.add_argument("--before", type=int, metavar="YEAR")
    cites.add_argument(
        "--most-cited", action="store_true", dest="most_cited",
        help="rank the most-cited verses instead",
    )
    cites.set_defaults(func=cmd_cites)

    spend = sub.add_parser("spend", help="what this has actually cost")
    spend.set_defaults(func=cmd_spend)

    cache = sub.add_parser("cache", help="inspect or clear the query cache")
    cache.add_argument("--clear", action="store_true", help="drop every cached answer")
    cache.add_argument("--prune", action="store_true", help="drop answers from older index builds")
    cache.set_defaults(func=cmd_cache)

    stats = sub.add_parser("stats", help="what's in the index")
    stats.set_defaults(func=cmd_stats)

    compact = sub.add_parser("compact", help="reclaim stranded vector rows")
    compact.set_defaults(func=cmd_compact)

    evaluate = sub.add_parser("eval", help="run the regression query set")
    evaluate.add_argument("--file", default=None)
    evaluate.add_argument("-n", type=int, default=10)
    evaluate.add_argument("--no-hyde", action="store_true")
    evaluate.add_argument("--no-rerank", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    serve = sub.add_parser("serve", help="local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if getattr(args, "max_per_doc", None) is None:
        args.max_per_doc = MAX_PER_DOC
    args.func(args)


if __name__ == "__main__":
    main()
