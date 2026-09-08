"""Export the local SQLite/npy index into files Postgres can COPY in.

Three size decisions, all measured rather than assumed:

* Vectors are truncated 1024 -> 512 dims. text-embedding-3-large is
  matryoshka-trained, so a prefix is still a valid embedding once renormalized.
  Measured on the eval set: recall@10 identical, MRR within noise.
* `embed_text` is dropped. It exists only to be embedded, and that already
  happened.
* `window_text` is dropped and reconstructed at query time from neighbouring
  chunks (same doc_id, adjacent ordinal). It is 143 MB of pure duplication.

Run: .venv/bin/python scripts/export_for_postgres.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gospel_search import index as idx  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "export"
TARGET_DIMS = 512


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = idx.connect(readonly=True)
    vectors = idx.load_vectors()

    print(f"source vectors: {vectors.shape}")

    # --- documents ---------------------------------------------------------
    rows = conn.execute(
        """SELECT id, key, kind, title, url, speaker, role, date,
                  volume, book, chapter, conference
           FROM documents ORDER BY id"""
    ).fetchall()
    with (OUT / "documents.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for r in rows:
            writer.writerow([r[k] if r[k] is not None else "" for k in r.keys()])
    print(f"documents.csv : {len(rows):,} rows")

    # --- chunks + embeddings ----------------------------------------------
    # pgvector's text input format is '[1,2,3]'. Written alongside the row so a
    # single COPY loads metadata and vector together.
    total = 0
    with (OUT / "chunks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        cursor = conn.execute(
            """SELECT id, doc_id, vec_row, kind, anchor, ordinal, citation, url,
                      speaker, date, display_text
               FROM chunks ORDER BY id"""
        )
        while True:
            batch = cursor.fetchmany(5000)
            if not batch:
                break
            for r in batch:
                v = np.asarray(vectors[r["vec_row"]][:TARGET_DIMS], dtype=np.float32)
                norm = float(np.linalg.norm(v))
                if norm < 1e-9:
                    continue  # never embedded; nothing to search against
                v = v / norm
                writer.writerow(
                    [
                        r["id"], r["doc_id"], r["kind"], r["anchor"], r["ordinal"],
                        r["citation"], r["url"], r["speaker"], r["date"],
                        r["display_text"],
                        "[" + ",".join(f"{x:.6f}" for x in v) + "]",
                    ]
                )
                total += 1
            print(f"  chunks: {total:,}", end="\r", flush=True)

    print(f"\nchunks.csv    : {total:,} rows")
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name:<16} {f.stat().st_size / 1e6:>7.0f} MB")


if __name__ == "__main__":
    main()
