/**
 * Run the same regression set as the Python `gospel eval`, against Postgres.
 *
 *   npm run eval               full pipeline
 *   npm run eval -- --raw      retrieval only (no HyDE, no rerank) — nearly free
 *
 * The point is to catch drift between the two implementations. Retrieval-only
 * numbers should match the Python side closely; the full-pipeline numbers will
 * wobble a little because HyDE and reranking are nondeterministic.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parse } from "yaml";

import { search, type Result } from "../src/lib/search";

type Case = {
  query: string;
  expect: string | string[];
  source?: "talks" | "scriptures";
};

const raw = process.argv.includes("--raw");
const file = join(import.meta.dirname, "..", "..", "tests", "queries.yaml");
const cases = (parse(readFileSync(file, "utf8")) as Case[]).filter(
  (c) => c?.query && c?.expect,
);

function matches(r: Result, expected: string[]): boolean {
  const hay = `${r.citation} ${r.title}`.toLowerCase();
  return expected.some((e) => hay.includes(e.toLowerCase()));
}

const N = 10;

async function main() {
  let hit1 = 0;
  let recall = 0;
  let mrr = 0;
  const misses: string[] = [];

  for (const c of cases) {
    const expected = Array.isArray(c.expect) ? c.expect : [c.expect];
    const results = await search(c.query, {
      n: N,
      filters: { source: c.source ?? null },
      useHyde: !raw,
      useRerank: !raw,
    });

    const idx = results.findIndex((r) => matches(r, expected));
    const rank = idx === -1 ? null : idx + 1;

    if (rank === 1) hit1++;
    if (rank) {
      recall++;
      mrr += 1 / rank;
    } else {
      misses.push(`${c.query}  ->  ${expected[0]}`);
    }

    const mark = rank === 1 ? "✓" : rank ? String(rank) : "✗";
    console.log(
      `  ${mark.padStart(3)}  ${c.query.slice(0, 64).padEnd(66)} -> ${expected[0]}`,
    );
  }

  const n = cases.length || 1;
  console.log(`\n  mode      ${raw ? "retrieval only" : "full pipeline"}`);
  console.log(`  cases     ${cases.length}`);
  console.log(`  hit@1     ${((hit1 / n) * 100).toFixed(0)}%`);
  console.log(`  recall@${N} ${((recall / n) * 100).toFixed(0)}%`);
  console.log(`  MRR       ${(mrr / n).toFixed(3)}`);
  if (misses.length) {
    console.log(`\n  missed entirely (${misses.length}):`);
    for (const m of misses) console.log(`    ${m}`);
  }

  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
