"""Command line entry point: build / search / eval / serve / stats."""

from __future__ import annotations

import argparse
import textwrap

from . import answer as answer_mod
from . import build as build_mod
from . import index as idx
from . import research as research_mod
from . import search as search_mod
from .config import MAX_PER_DOC
from .plan import Plan, plan as make_plan

BOLD, DIM, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"


def wrap(text: str, width: int = 88, indent: str = "     ") -> str:
    return textwrap.fill(
        text, width=width, initial_indent=indent, subsequent_indent=indent
    )


def cmd_build(args) -> None:
    conn = idx.connect()
    idx.init(conn)

    if args.only in (None, "scriptures"):
        build_mod.build_scriptures(
            conn, summaries=args.summaries, verify=not args.no_verify
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
    if args.no_plan:
        plan = Plan(topic=args.query, filters=filters)
    else:
        plan = make_plan(args.query, override=filters)
        if args.explain:
            print(f"{DIM}     plan: {plan.describe()}{RESET}")

    findings = research_mod.investigate(
        args.query,
        plan,
        n=args.n,
        use_hyde=not args.no_hyde,
        use_rerank=not args.no_rerank,
    )
    results = findings.results

    if not results:
        print("No matches.")
        return

    if not args.no_answer:
        text = answer_mod.answer(args.query, findings)
        print(f"\n{BOLD}{text}{RESET}\n")
        print(DIM + "─" * 88 + RESET)

    # What the answer actually saw. Without this the output reads like a survey
    # of the whole corpus regardless of how thin the evidence was.
    line = findings.coverage.summary()
    if findings.exhaustive:
        line = f"complete: all {findings.total_matches} matches · " + line
    print(f"{DIM}     [{plan.intent}] {line}{RESET}")

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
    build.add_argument(
        "--summaries", action="store_true", help="also index chapter head-notes"
    )
    build.add_argument("--no-embed", action="store_true", help="ingest only")
    build.add_argument("--no-verify", action="store_true", help="skip the URL check")
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
    search.set_defaults(func=cmd_search)

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
