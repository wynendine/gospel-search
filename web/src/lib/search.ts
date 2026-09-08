import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

import { query } from "@/lib/db";

// Mirrors src/gospel_search/config.py. Kept in sync by hand; the eval script
// is what catches a drift between the two implementations.
export const EMBED_MODEL = "text-embedding-3-large";
export const EMBED_DIMS = 512; // matryoshka truncation — measured lossless here
export const ANSWER_MODEL = "claude-opus-5";
export const RERANK_MODEL = "claude-opus-5";
export const HYDE_MODEL = "claude-opus-5";

const DENSE_K = 100;
const LEXICAL_K = 100;
const RRF_K = 60;
const RERANK_K = 50;
export const ANSWER_K = 12;

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export type Filters = {
  speaker?: string | null;
  after?: number | null;
  before?: number | null;
  source?: "talks" | "scriptures" | null;
};

export type Result = {
  chunkId: number;
  citation: string;
  url: string;
  speaker: string;
  date: string;
  kind: string;
  title: string;
  displayText: string;
  windowText: string;
  denseRank: number | null;
  lexicalRank: number | null;
  fused: number;
  rerank: number | null;
};

/** Build the shared WHERE fragment plus its bound parameters. */
function filterSql(f: Filters, from = 1): { sql: string; params: unknown[] } {
  const clauses: string[] = [];
  const params: unknown[] = [];
  let i = from;

  if (f.speaker) {
    clauses.push(`c.speaker ILIKE $${i++}`);
    params.push(`%${f.speaker}%`);
  }
  if (f.after) {
    clauses.push(`c.date >= $${i++}`);
    params.push(`${f.after}-01-01`);
  }
  if (f.before) {
    clauses.push(`c.date <= $${i++}`);
    params.push(`${f.before}-12-31`);
  }
  if (f.source === "talks") clauses.push(`c.kind = 'talk'`);
  else if (f.source === "scriptures") clauses.push(`c.kind IN ('verse','summary')`);

  return { sql: clauses.length ? clauses.join(" AND ") : "TRUE", params };
}

// --- Query expansion -------------------------------------------------------

const HYDE_PROMPT = `You help search a corpus of General Conference talks and the \
standard works (Bible, Book of Mormon, Doctrine and Covenants, Pearl of Great Price).

The user is trying to find a passage they half-remember. Write 2-3 sentences of \
the passage they are most likely looking for, in the voice and register of the \
source — a conference talk or scripture, as fits the query. Do not answer the \
question, do not hedge, do not mention the search. Just write the passage as it \
would plausibly appear.

Query: `;

export async function hyde(q: string): Promise<string> {
  const res = await anthropic.messages.create({
    model: HYDE_MODEL,
    max_tokens: 400,
    output_config: { effort: "low" },
    messages: [{ role: "user", content: HYDE_PROMPT + q }],
  });
  return res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("")
    .trim();
}

export async function embed(texts: string[]): Promise<number[][]> {
  const res = await openai.embeddings.create({
    model: EMBED_MODEL,
    input: texts,
    dimensions: EMBED_DIMS,
  });
  return res.data.map((d) => {
    const v = d.embedding;
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  });
}

// --- Retrieval halves ------------------------------------------------------

async function dense(vec: number[], f: Filters, k = DENSE_K) {
  const { sql, params } = filterSql(f, 2);
  // <=> is cosine distance in pgvector; vectors are stored normalized so this
  // orders identically to the local numpy dot product.
  const rows = await query<{ id: number; score: number }>(
    `SELECT c.id, 1 - (c.embedding <=> $1::halfvec) AS score
       FROM chunks c
      WHERE ${sql}
      ORDER BY c.embedding <=> $1::halfvec
      LIMIT ${k}`,
    [JSON.stringify(vec), ...params],
  );
  return rows.map((r) => [r.id, Number(r.score)] as [number, number]);
}

async function lexical(q: string, f: Filters, k = LEXICAL_K) {
  const { sql, params } = filterSql(f, 2);
  // websearch_to_tsquery tolerates whatever a person types — quotes, OR, minus
  // — without the escaping dance the SQLite FTS5 version needed.
  const rows = await query<{ id: number; score: number }>(
    `SELECT c.id,
            ts_rank_cd(to_tsvector('english', c.display_text || ' ' || c.citation),
                       websearch_to_tsquery('english', $1)) AS score
       FROM chunks c
      WHERE to_tsvector('english', c.display_text || ' ' || c.citation)
            @@ websearch_to_tsquery('english', $1)
        AND ${sql}
      ORDER BY score DESC
      LIMIT ${k}`,
    [q, ...params],
  );
  return rows.map((r) => [r.id, Number(r.score)] as [number, number]);
}

// --- Fusion ----------------------------------------------------------------

function fuse(
  denseHits: [number, number][],
  lexicalHits: [number, number][],
  k = RRF_K,
) {
  const scores = new Map<number, number>();
  const ranks = new Map<number, { dense?: number; lexical?: number }>();

  for (const [label, hits] of [
    ["dense", denseHits],
    ["lexical", lexicalHits],
  ] as const) {
    hits.forEach(([id], i) => {
      scores.set(id, (scores.get(id) ?? 0) + 1 / (k + i + 1));
      ranks.set(id, { ...(ranks.get(id) ?? {}), [label]: i + 1 });
    });
  }

  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id, score]) => ({ id, score, rank: ranks.get(id)! }));
}

/** Hydrate ids into results, rebuilding window text from adjacent chunks. */
async function hydrate(
  fused: { id: number; score: number; rank: { dense?: number; lexical?: number } }[],
): Promise<Result[]> {
  if (!fused.length) return [];
  const ids = fused.map((f) => f.id);

  // window_text isn't stored — it's 143 MB of duplication. Rebuild it by
  // pulling each hit's neighbours by (doc_id, ordinal). Scripture verses join
  // with a space, talk paragraphs with a blank line, matching chunk.py.
  const rows = await query<{
    id: number;
    citation: string;
    url: string;
    speaker: string;
    date: string;
    kind: string;
    display_text: string;
    title: string;
    window_text: string;
  }>(
    `WITH hit AS (SELECT * FROM chunks WHERE id = ANY($1::int[]))
     SELECT h.id, h.citation, h.url, h.speaker, h.date, h.kind,
            h.display_text, d.title,
            COALESCE((
              SELECT string_agg(n.display_text,
                                CASE WHEN h.kind = 'talk' THEN E'\\n\\n' ELSE ' ' END
                                ORDER BY n.ordinal)
                FROM chunks n
               WHERE n.doc_id = h.doc_id
                 AND n.kind = h.kind
                 AND n.ordinal BETWEEN h.ordinal - 1 AND h.ordinal + 1
            ), h.display_text) AS window_text
       FROM hit h JOIN documents d ON d.id = h.doc_id`,
    [ids],
  );

  const byId = new Map(rows.map((r) => [r.id, r]));
  return fused.flatMap(({ id, score, rank }) => {
    const r = byId.get(id);
    if (!r) return [];
    return [
      {
        chunkId: id,
        citation: r.citation,
        url: r.url,
        speaker: r.speaker,
        date: r.date,
        kind: r.kind,
        title: r.title,
        displayText: r.display_text,
        windowText: r.window_text,
        denseRank: rank.dense ?? null,
        lexicalRank: rank.lexical ?? null,
        fused: score,
        rerank: null,
      },
    ];
  });
}

// --- Rerank ----------------------------------------------------------------

const RERANK_PROMPT = `A person is searching General Conference talks and the \
standard works for a passage they half-remember. Score how well each numbered \
passage answers their query.

Query: {query}

Score 0-10:
  9-10  this is almost certainly the passage they mean
  6-8   directly on the subject, plausibly what they want
  3-5   related subject, probably not the passage
  0-2   not relevant

Judge the passage's own content. A passage is not more relevant because it \
repeats the query's words, and not less relevant because it uses none of them.

Passages:
{passages}`;

export async function rerank(q: string, results: Result[], k = RERANK_K) {
  const candidates = results.slice(0, k);
  if (!candidates.length) return results;

  const passages = candidates
    .map((r, i) => `[${i}] ${r.citation}\n${r.displayText.slice(0, 700)}`)
    .join("\n\n");

  const res = await anthropic.messages.create({
    model: RERANK_MODEL,
    max_tokens: 4000,
    output_config: {
      effort: "low",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            scores: {
              type: "array",
              items: {
                type: "object",
                properties: { id: { type: "integer" }, score: { type: "number" } },
                required: ["id", "score"],
                additionalProperties: false,
              },
            },
          },
          required: ["scores"],
          additionalProperties: false,
        },
      },
    },
    messages: [
      {
        role: "user",
        content: RERANK_PROMPT.replace("{query}", q).replace("{passages}", passages),
      },
    ],
  });

  const text = res.content.find((b) => b.type === "text");
  const parsed = JSON.parse(text && "text" in text ? text.text : "{}") as {
    scores?: { id: number; score: number }[];
  };
  const scores = new Map((parsed.scores ?? []).map((s) => [s.id, s.score]));

  candidates.forEach((r, i) => {
    r.rerank = scores.get(i) ?? 0;
  });
  candidates.sort((a, b) => (b.rerank ?? 0) - (a.rerank ?? 0));
  return [...candidates, ...results.slice(k)];
}

// --- Entry point -----------------------------------------------------------

export async function search(
  q: string,
  opts: {
    n?: number;
    filters?: Filters;
    useHyde?: boolean;
    useRerank?: boolean;
  } = {},
): Promise<Result[]> {
  const { n = 10, filters = {}, useHyde = true, useRerank = true } = opts;

  const texts = [q];
  if (useHyde) {
    try {
      const h = await hyde(q);
      if (h) texts.push(h);
    } catch {
      // HyDE is an optimization, not a gate — fall back to the raw query.
    }
  }

  const vectors = await embed(texts);
  const mean = vectors[0].map(
    (_, i) => vectors.reduce((s, v) => s + v[i], 0) / vectors.length,
  );
  const norm = Math.hypot(...mean) || 1;
  const qvec = mean.map((x) => x / norm);

  const [denseHits, lexicalHits] = await Promise.all([
    dense(qvec, filters),
    lexical(q, filters),
  ]);

  let results = await hydrate(fuse(denseHits, lexicalHits));
  if (useRerank && results.length) results = await rerank(q, results);
  return results.slice(0, n);
}
