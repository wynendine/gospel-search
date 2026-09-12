# Gospel Search

Ask a question about General Conference talks (1971–present) and the standard
works, and get an answer built from actual passages — speaker, date, and a link
to the exact paragraph on churchofjesuschrist.org.

The problem it solves: you remember the *idea* but not the words, and keyword
search wants the words.

```
$ gospel search "a passage about God letting people choose rather than compelling them"

The closest match is Moses 3:17, where the Lord forbids the tree of knowledge
but adds, "nevertheless, thou mayest choose for thyself, for it is given unto
thee" [3]. For a broader statement of agency, 2 Nephi 2:27 teaches that men are
"free to choose liberty and eternal life" [2], and Helaman 14:30 that "ye are
free; ye are permitted to act for yourselves" [1].

[1] Helaman 14:30                                  rerank 9/10 · dense
    …ye are free; ye are permitted to act for yourselves…
    https://www.churchofjesuschrist.org/study/scriptures/bofm/hel/14?lang=eng#p30
```

None of those verses share a meaningful word with the query. That's the point.

## What it can answer

Different questions need different retrieval, so the question is classified
first. Running one strategy over all of them is how "every reference to X"
quietly becomes "the ten best passages about X".

| you ask | it does |
|---|---|
| "the talk about the father whose son had a rare illness" | ranked passage search |
| "every reference to charity in the Book of Mormon" | exact term match, complete, in canonical order |
| "compare Bednar and Holland on faith" | one scoped search per person |
| "what has been taught about the gathering of Israel" | six facet searches, merged |
| "which talks cite Alma 32:21" | citation graph lookup |
| "D&C 121:7" | exact verse, no search at all |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Two keys in `.env` at the project root:

```
OPENAI_API_KEY=...      # embeddings
ANTHROPIC_API_KEY=...   # planning, query expansion, reranking, answers
```

## Building the index

```bash
gospel build --only scriptures     # ~1 min crawl, ~4 min embed, ~$0.65
gospel build --only talks          # ~20 min crawl, ~27 min embed, ~$4.45
gospel build --summaries           # + chapter head-notes, ~7 min, ~$0.01
gospel build --citations           # + citation graph, from cached pages, free
```

Measured on the real corpus: **4,254 talks (April 1971 – April 2026) and 41,995
verses become 190,100 chunks**, ~39M tokens, **~$5.10 one time**, 1.4 GB on disk.

Every stage is resumable and cached. Re-running `build` after it finishes costs
nothing — chunks are matched by anchor, so an unchanged one keeps its id and
therefore its vector. Interrupt it whenever; it picks up where it stopped.

Crawling runs 5 workers behind one shared clock for a flat ~4 requests/second —
about what a browser does opening one page. `FETCH_WORKERS` and `REQUEST_DELAY`
in `config.py` set the rate; fetching is split from parsing so the network work
happens all at once.

## Searching

```bash
gospel search "the talk about the father whose son had a rare illness"
gospel search "every reference to charity in the Book of Mormon"
gospel search "compare Bednar and Holland on faith" --explain
gospel search "covenant belonging" --speaker Nelson --after 2018
gospel search "Alma 32:21"                        # exact verse, instant, free
gospel serve                                      # web UI at localhost:8000
```

Scoping can be written into the question — "in the Book of Mormon", "since
2015", "what did Nelson say" — and the planner extracts it. Explicit flags
always win over what it infers.

Flags: `--explain` (print the plan), `--no-plan`, `--no-answer` (passages only),
`--no-rerank` / `--no-hyde` (drop the LLM stages — much faster, noticeably
worse), `--source`, `--volume`, `--book`, `--speaker`, `--after` / `--before`,
`--max-per-doc`, `-n`.

Other commands: `gospel cites`, `gospel stats`, `gospel compact`, `gospel eval`.

## How it works

```
question
  ├─ reference? ───── "D&C 121:7" resolves directly. No plan, no embedding, no cost.
  └─ plan ─────────── Claude picks an intent and pulls scoping out of the prose
       ├─ lookup ──── find one half-remembered passage    → rank, top-k
       ├─ enumerate ─ occurrences of a term               → exact, complete, counted
       ├─ compare ─── contrast two or more speakers       → one search each
       ├─ thematic ── survey a subject                    → one search per facet
       └─ citations ─ who quotes a verse                  → graph lookup

retrieval (lookup / compare / thematic)
  ├─ HyDE ─────────── Claude writes the passage you probably mean
  ├─ dense ────────── query + HyDE embedded, averaged, matmul over all vectors
  ├─ BM25 ─────────── SQLite FTS5 over the same chunks
  ├─ RRF ──────────── fuse the two ranked lists
  ├─ rerank ───────── Claude scores the top 50 for actual relevance
  ├─ diversify ────── cap passages per talk/chapter so one can't take every slot
  └─ answer ───────── cited summary written for the intent; passages below
```

**Chunking does most of the work.** Each chunk is a 3-unit sliding window
(paragraphs for talks, verses for scripture) but is *cited and displayed* as its
center unit — big enough to embed meaningfully, precise enough to link to. Each
window is prefixed with its provenance before embedding, so the speaker, title,
and date live inside the vector. That's why "what did Nelson say about
covenants" works without a `--speaker` flag.

**Both retrieval halves earn their place.** Dense alone fumbles exact names and
quoted phrases; BM25 alone is the keyword guessing this exists to replace.
Results are tagged `dense`, `bm25`, or `dense+bm25` so you can see which half
found a passage. Stopwords are dropped from the lexical query — terms are OR-ed,
so leaving them in made BM25's top hits share no content word with the question
(hit@1 45% → 59%).

**Reranking is the biggest quality lever after chunking.** The fused top-10 and
the reranked top-10 are routinely different lists.

**Comparison retrieves per person.** A single blended query for "compare Bednar
and Holland on faith" returns passages about faith, most by neither man — 3 of
10 in testing. One scoped search each gives 16 passages from 16 talks spanning
1996–2024.

**Thematic surveys retrieve per facet.** One broad query embeds to the centre of
a subject and returns that centre. On "the gathering of Israel" a single query
found 12 sources; six facets — temple work, missionary work, the scattering,
covenant and adoption, who participates, gathering to Zion — surfaced 39. The
single query was seeing 31% of the material.

**Enumeration is exact, or honest about not being.** "Every reference to charity
in the Book of Mormon" returns all 22 matches in canonical order and says so.
Three things it gets right that are easy to get wrong:

- *Too many to list.* "All references to faith in conference" matches 15,188.
  Past the cap the answer becomes a **distribution** — by decade for talks, by
  book for scripture — plus the strongest examples, saying plainly that it is
  not an index and what narrowing would produce one.
- *Together vs. either.* "Faith and repentance together in the Book of Mormon"
  intersects rather than unions: 39 passages, not 572.
- *Stemming.* Matching is stemmed, so "faith" also catches "faithful" (16,081
  chunks against 13,455 literal). Usually wanted — and said, rather than
  claiming a completeness it doesn't have.

## The citation graph

Talks footnote the verses they quote, and those footnotes are structured links.
Parsing them turns the corpus into a graph: **77,859 talk→scripture links from
3,822 of 4,254 talks**, and 77,844 of them record the *paragraph* doing the
citing, not just the talk.

```bash
gospel cites "Alma 32:21"             # which talks quote it, and the sentence
gospel cites "D&C 121" --after 2015
gospel cites --most-cited -n 20       # most-quoted verses in conference
gospel search "which talks cite Moses 1:39"   # same thing, in prose
```

Footnotes only exist from about 1990, so earlier talks are covered by a second
pass over the prose, where references were written inline — "(D&C 64:9)". That
pass only accepts a match whose book resolves through the reference parser's
alias table, which keeps "the meeting ran 10:30 to 11:45" out of the graph.
43,969 links come from footnotes, 33,890 from prose.

The graph answers what search cannot, because it knows what a talk *quotes*
rather than what it is *about*. Most-cited across the corpus: Moses 1:39 (233
talks), Mosiah 18:9 (158), 2 Nephi 31:20 (150). Restricted to the 2020s it is
Alma 7:11–12 — a genuinely different answer.

## Storage

Vectors live in a plain memory-mapped `.npy`, not a vector database —
`sqlite-vec` can't load in this Python (no `enable_load_extension`). A
brute-force matmul over the full corpus measures **15 ms warm**, which is
*exact* and needs no index tuning, so nothing was given up. First query after
start pays ~1.2 s to page the file in. SQLite holds metadata, the FTS5 index,
and the citation graph.

```
data/index/corpus.db     metadata + full-text index + citations
data/index/vectors.npy   float32, L2-normalized, row i == chunk id i+1
data/raw/                cached API responses — delete to force a refetch
```

Filtering is vectorized against dictionary-encoded columns rather than pulled
through SQLite, so `--source talks` costs the same as no filter (it used to cost
8× more). `gospel compact` reclaims vector rows stranded by re-ingesting a
changed document, carrying existing vectors across so it never re-embeds.

## Tuning

`gospel eval` runs `tests/queries.yaml` and reports hit@1, recall@10, and MRR.

| | hit@1 | recall@10 | MRR |
|---|---|---|---|
| full pipeline | **91%** | **100%** | **~0.95** |
| `--no-hyde --no-rerank` | 59% | 95% | 0.703 |

The second row costs almost nothing to run and is the right way to judge a
change to chunking or fusion — but it is a diagnostic, not a target. Paraphrase
questions are *expected* to fail there; recovering them is precisely what HyDE
and reranking do. Expect small run-to-run variation in the full pipeline: both
are LLM calls, so a borderline case trades places between rank 1 and 2.

Add a case every time it misses something you knew was there. Every real
improvement in this project came from a measurement contradicting an
assumption — phrases already worked when I thought they didn't; the eval case I
chased turned out not to be a bug, but diagnosing it found a stopword problem
worth 14 points of hit@1.

Knobs live in `src/gospel_search/config.py`: retrieval depths, the RRF
constant, window size, per-document caps, and a model per stage. Every model
and the rerank depth can be overridden per run with an env var, which is how
the numbers below were measured:

```bash
GOSPEL_RERANK_MODEL=claude-haiku-4-5 gospel eval
```

## Cost

A search is four Claude calls. Measured per query:

| stage | cost | share |
|---|---|---|
| plan | $0.0103 | 9% |
| HyDE | $0.0064 | 6% |
| rerank | $0.0605 | 53% |
| answer | $0.0369 | 32% |
| **total** | **$0.114** | |

References (`D&C 121:7`) and `gospel cites` are database lookups — instant and
free. Repeated queries are cached and also free; `gospel cache` inspects it and
`--fresh` bypasses it.

What each stage is worth, measured on the eval set rather than assumed:

| configuration | hit@1 | MRR | cost |
|---|---|---|---|
| all Opus 5 | 91% | 0.947 | $0.114 |
| **Haiku rerank** | **91%** | **0.941** | **$0.066** |
| Haiku rerank + plan | 86% | 0.909 | $0.058 |
| Haiku rerank + plan + HyDE | 77% | 0.873 | $0.055 |
| Haiku rerank, 25 candidates, 450 chars | — | 0.873 | $0.050 |

**Reranking is free to downgrade** — 0.006 MRR is inside the run-to-run noise
of two identical Opus runs, and it cuts the bill 42%. Planning and HyDE are
not: HyDE writes a passage in scriptural register, which is a generation task,
and cutting the rerank pool to 25 loses a case from recall entirely. Synthesis
was never tested cheaper because it is the part you actually read.

## Layout

```
src/gospel_search/
  sources/          study-API client, conference crawler, scriptures CSV
  chunk.py          windowing + provenance prefixes
  index.py          SQLite schema, FTS5, memory-mapped vectors
  embed.py          OpenAI embeddings, resumable
  plan.py           question -> intent + scoping
  research.py       the per-intent retrieval strategies
  search.py         HyDE, dense, BM25, RRF, rerank, diversify
  reference.py      exact scripture reference resolution
  citations.py      citation extraction and graph queries
  answer.py         per-intent synthesis
  cli.py / web.py   terminal and local web UI
tests/
  test_units.py     98 checks, no network or API keys needed
  queries.yaml      the retrieval regression set
web/                Next.js + Neon port (deploy currently parked)
```

## A note on the text

General Conference talks are © Intellectual Reserve. This builds a local,
personal-study index and links back to the official source rather than
substituting for it. The repository holds no corpus text — `data/` is
gitignored and `gospel build` regenerates it from public sources. Worth
rethinking before putting a built index on a public URL.
