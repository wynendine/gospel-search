# Gospel Search

Semantic search over General Conference talks (1971–present) and the standard
works. Describe what was said; it finds where it was said — with the speaker,
the date, and a link to that exact paragraph on churchofjesuschrist.org.

The problem it solves: you remember the *idea* but not the words, and keyword
search wants the words.

```
$ gospel search "a passage about God letting people choose rather than compelling"

The closest match is Moses 3:17, where the Lord forbids the tree of knowledge
but adds, "nevertheless, thou mayest choose for thyself" [3]. …

[1] Helaman 14:30                    rerank 9/10 · dense
    …ye are free; ye are permitted to act for yourselves…
    https://www.churchofjesuschrist.org/study/scriptures/bofm/hel/14?lang=eng#p30
```

None of those verses share meaningful words with the query. That's the point.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Put two keys in `.env` at the project root:

```
OPENAI_API_KEY=...      # embeddings
ANTHROPIC_API_KEY=...   # query expansion, reranking, answers
```

## Building the index

```bash
gospel build --only scriptures     # ~1 min crawl, ~4 min embed, ~$0.65
gospel build --only talks          # ~20 min crawl, ~27 min embed, ~$4.45
gospel build                       # both
```

Measured on the real corpus: 4,254 talks (April 1971 – April 2026) and 41,995
verses become 190,100 chunks (including chapter head-notes), ~39M tokens,
**~$5.10 one time**, 1.4 GB on disk.

Every stage is resumable and cached. Re-running `build` after it finishes costs
nothing — unchanged documents keep their ids, their vectors, and their place.
Interrupt it whenever; it picks up where it stopped.

Crawling runs 5 workers behind one shared clock, for a flat ~4 requests/second
no matter the worker count — about what a browser does opening a single page.
`FETCH_WORKERS` and `REQUEST_DELAY` in `config.py` set the rate. Fetching is
split from parsing so the network work all happens at once.

`--summaries` also indexes chapter head-notes — the "Alma compares the word unto
a seed" abstracts. They answer "which chapter is about X" with the *chapter*
rather than a verse buried inside it, and they rank first for that shape of
query. 1,580 chunks, ~7 min of crawling, ~$0.01.

Head-notes are read from `p.study-summary`, falling back to `p.study-intro`
(short D&C sections carry only the historical header) and then `p.subtitle`
(Joseph Smith—History). Two chapters have nothing to extract and get no summary
chunk: Psalms 119 and Articles of Faith 1.

Adding them later is cheap on purpose: chunks are matched by anchor, so the
41,995 verse chunks keep their ids and their vectors, and only the new summary
chunks get embedded.

## Searching

```bash
gospel search "the talk about the father whose son had a rare illness"
gospel search "covenant belonging" --speaker Nelson --after 2018
gospel search "faith like a seed" --source scriptures --no-answer
gospel serve                       # web UI at localhost:8000
```

```bash
gospel search "every reference to charity in the Book of Mormon"
gospel search "compare Bednar and Holland on faith" --explain
```

Useful flags: `--no-answer` (passages only), `--no-rerank` and `--no-hyde` (drop
the LLM stages — much faster, noticeably worse), `--explain` (show the plan),
`--no-plan`, `--source`, `--volume`, `--book`, `--speaker`, `--after` /
`--before`, `--max-per-doc`, `-n`.

Scoping can be written into the question — "in the Book of Mormon", "since
2015", "what did Nelson say" — and the planner extracts it. Explicit flags
always win over what it infers.

## How it works

A question is first *classified*, because different questions need different
retrieval. Applying one strategy to all of them is how "every reference to X"
quietly becomes "the ten best passages about X".

```
question
  └─ plan ─────────── Claude picks an intent and pulls scoping out of the prose
       ├─ lookup ──── find one half-remembered passage      → rank, top-k
       ├─ enumerate ─ list every occurrence of a term       → exact, complete
       ├─ compare ─── contrast two or more speakers         → one search each
       └─ thematic ── survey teaching on a subject          → breadth, 1/source

retrieval (lookup / compare / thematic)
  ├─ HyDE ─────────── Claude writes the passage you probably mean
  ├─ dense ────────── query + HyDE embedded, averaged, matmul over all vectors
  ├─ BM25 ─────────── SQLite FTS5 over the same chunks
  ├─ RRF ──────────── fuse the two ranked lists
  ├─ rerank ───────── Claude scores the top 50 for actual relevance
  ├─ diversify ────── cap passages per talk/chapter so one can't take every slot
  └─ answer ───────── cited summary, written for the intent; passages below
```

`--explain` prints the plan. `--no-plan` skips it and treats the query as a
plain lookup.

**Enumeration is exact.** "Every reference to charity in the Book of Mormon"
returns all 22 matches in canonical order and says so; it is a FTS term match
with a real count, not a ranking. For a *concept* rather than a fixed word the
answer says explicitly that it is not a complete list, because it can't be.

**Comparison retrieves per person.** A single blended query for "compare Bednar
and Holland on faith" returns passages about faith, most by neither man — only
3 of 10 in testing. Scoping one search per speaker gives 16 passages from 16
talks spanning 1996-2024.

**Thematic surveys retrieve per facet.** One broad query embeds to the centre of
a subject and returns that centre. Measured on "the gathering of Israel": a
single query found 12 sources, while the same question split into six facets —
temple work, missionary work, the scattering, covenant and adoption, who
participates, gathering to Zion — surfaced 39. The single query was seeing 31%
of the material. Facets are retrieved without HyDE or reranking (they are
already specific), then one rerank pass orders the union.

**Chunking** does most of the work. Each chunk is a 3-unit sliding window
(paragraphs for talks, verses for scripture) but is *cited and displayed* as its
center unit — big enough to embed meaningfully, precise enough to link to. Each
window is prefixed with its provenance before embedding, so the speaker, title,
and date live inside the vector. That's why `--speaker`-free queries like "what
did Nelson say about covenants" still work.

**Both retrieval halves earn their place.** Dense alone fumbles exact names and
quoted phrases; BM25 alone is the keyword guessing this tool exists to replace.
Results are tagged `dense`, `bm25`, or `dense+bm25` so you can see which half
found a given passage.

**Reranking** is the biggest quality lever after chunking. The fused top-10 and
the reranked top-10 are routinely different lists.

## The citation graph

Talks footnote the verses they quote, and those footnotes are structured links.
Parsing them turns the corpus into a graph — **77,859 talk→scripture links from
3,822 of 4,254 talks**, each recording the paragraph that does the citing.

```bash
gospel build --citations              # from cached pages; no refetch, no re-embed
gospel cites "Alma 32:21"             # which talks quote it, and the sentence
gospel cites "D&C 121" --after 2015
gospel cites --most-cited -n 20       # most-quoted verses in conference
gospel search "which talks cite Moses 1:39"   # same thing in prose
```

Footnotes only exist from about 1990, so talks before that are covered by a
second pass over the prose, where references were written inline — "(D&C 64:9)".
That pass only accepts a match whose book name resolves through the reference
parser's alias table, which is what keeps "the meeting ran 10:30 to 11:45" out
of the graph. 42,845 links come from footnotes, 35,014 from prose.

The graph answers questions search cannot, because it knows what a talk
*quotes* rather than what it is *about*. Most-cited across the corpus:
Moses 1:39 (233 talks), Mosiah 18:9 (158), 2 Nephi 31:20 (150).

## Storage

Vectors live in a plain memory-mapped `.npy`, not a vector database —
`sqlite-vec` can't load in this Python (no `enable_load_extension`). At full
corpus scale a brute-force matmul measures **23 ms**, which is *exact* and needs
no index tuning, so nothing was given up. SQLite holds metadata and the FTS5
index (BM25 runs ~150 ms, so it, not the vectors, is the slower half).

```
data/index/corpus.db     metadata + full-text index
data/index/vectors.npy   float32, L2-normalized, row i == chunk id i+1
data/raw/                cached API responses — delete to force a refetch
```

`gospel compact` reclaims vector rows stranded by re-ingesting a changed
document. It carries existing vectors across, so it never re-embeds.

## Tuning

`gospel eval` runs `tests/queries.yaml` and reports hit@1, recall@10, and MRR.
The baseline on the full corpus is **hit@1 91%, recall@10 100%, MRR 0.955**
across 22 cases. Expect small run-to-run variation: HyDE and reranking are both
LLM calls, so a borderline case can trade places between rank 1 and 2.
Add a case every time the tool misses something you knew was there — that's what
keeps tuning honest rather than vibes-based. Compare configurations with
`--no-rerank` / `--no-hyde` to see what each stage is actually buying.

Knobs live in `src/gospel_search/config.py`: retrieval depths, the RRF constant,
window size, and model choices. Reranking is the step to point at
`claude-haiku-4-5` if per-query latency starts to annoy you.

## A note on the text

General Conference talks are © Intellectual Reserve. This builds a local,
personal-study index and links back to the official source rather than
substituting for it. Worth rethinking before putting it on a public URL.
