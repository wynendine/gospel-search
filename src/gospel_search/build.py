"""Ingest -> chunk -> embed -> index. Resumable at every stage."""

from __future__ import annotations

import sys

from . import chunk as chunker
from . import embed
from . import index as idx
from .sources import church_api, conference, scriptures


def build_scriptures(conn, *, summaries: bool = False, verify: bool = True) -> int:
    print("Downloading the scriptures CSV...")
    scriptures.download_csv()

    chapters = list(scriptures.iter_chapters())
    print(f"  {len(chapters)} chapters, {sum(len(c.verses) for c in chapters)} verses")

    if verify:
        print("Verifying one URL per volume against the live site...")
        failures = scriptures.verify_slugs(chapters)
        if failures:
            raise SystemExit("URL slugs do not resolve:\n  " + "\n  ".join(failures))
        print("  all volumes resolve")

    if summaries:
        print(f"Fetching {len(chapters)} chapter summaries...")
        church_api.prefetch([c.uri for c in chapters], progress=_bar("chapters"))
        print()
        for chapter in chapters:
            chapter.summary = scriptures.fetch_summary(chapter)

    total = 0
    for chapter in chapters:
        document, chunks = chunker.chunk_chapter(chapter)
        total += idx.add_document(conn, document, chunks)
    conn.commit()
    print(f"  {total} scripture chunks")
    return total


def _bar(label):
    def progress(done, total):
        if total:
            sys.stdout.write(f"\r  {label}: {done}/{total}")
            sys.stdout.flush()

    return progress


def build_talks(conn, *, periods=None) -> int:
    periods = periods or conference.conference_periods()

    # Fetching dominates; parsing is milliseconds. So pull everything
    # concurrently first, then walk the cache. Two passes are needed because
    # talk URIs are only discoverable from the manifests.
    manifests = [f"/general-conference/{year}/{month}" for year, month in periods]
    print(f"Fetching {len(manifests)} conference manifests...")
    church_api.prefetch(manifests, progress=_bar("manifests"))
    print()

    per_period = {}
    all_uris = []
    for year, month in periods:
        uris = conference.talk_uris(year, month)
        per_period[(year, month)] = uris
        all_uris.extend(uris)

    print(f"Fetching {len(all_uris)} talk pages...")
    church_api.prefetch(all_uris, progress=_bar("talks"))
    print()

    print("Parsing and chunking...")
    total = talks = 0
    for year, month in periods:
        found = 0
        for uri in per_period[(year, month)]:
            talk = conference.parse_talk(uri, year, month)
            if talk is None:
                continue
            document, chunks = chunker.chunk_talk(talk)
            total += idx.add_document(conn, document, chunks)
            found += 1
        talks += found
        conn.commit()
    print(f"  {talks} talks, {total} new chunks")
    return total


def finalize(conn, *, skip_embed: bool = False) -> None:
    before, after = idx.compact(conn)
    if before != after:
        print(f"Compacted {before - after} stranded vector rows ({before} -> {after})")

    rows = idx.assign_vector_rows(conn)
    print(f"Vector rows: {rows}")

    pending = embed.pending_count(conn)
    if pending and not skip_embed:
        tokens, cost = embed.estimate_cost(conn)
        print(f"Embedding {pending} chunks (~{tokens:,} tokens, ~${cost:.2f})...")

        def progress(done):
            pct = 100 * done / pending
            sys.stdout.write(f"\r  {done}/{pending} ({pct:.1f}%)")
            sys.stdout.flush()

        embed.embed_pending(conn, progress=progress)
        print()
    elif pending:
        print(f"Skipping embedding ({pending} chunks pending)")
    else:
        print("All chunks already embedded")

    print("Rebuilding the full-text index...")
    idx.rebuild_fts(conn)
    idx.write_meta(conn, rows)
    print("Done.")
