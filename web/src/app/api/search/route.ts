import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/auth";
import { answer } from "@/lib/answer";
import { ANSWER_K, search } from "@/lib/search";

// HyDE + rerank + synthesis is three sequential Claude calls; ~20s is normal
// and the Hobby ceiling is 60s.
export const maxDuration = 60;
export const runtime = "nodejs";

const Body = z.object({
  query: z.string().min(2).max(500),
  n: z.number().int().min(1).max(50).default(10),
  speaker: z.string().max(100).nullable().default(null),
  after: z.number().int().min(1830).max(2200).nullable().default(null),
  before: z.number().int().min(1830).max(2200).nullable().default(null),
  source: z.enum(["talks", "scriptures"]).nullable().default(null),
  answer: z.boolean().default(true),
  hyde: z.boolean().default(true),
  rerank: z.boolean().default(true),
});

export async function POST(request: Request) {
  const session = await auth();
  if (!session) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const parsed = Body.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "bad request" }, { status: 400 });
  }
  const body = parsed.data;

  try {
    const results = await search(body.query, {
      n: body.n,
      filters: {
        speaker: body.speaker,
        after: body.after,
        before: body.before,
        source: body.source,
      },
      useHyde: body.hyde,
      useRerank: body.rerank,
    });

    const text =
      body.answer && results.length
        ? await answer(body.query, results.slice(0, ANSWER_K))
        : "";

    return NextResponse.json({
      answer: text,
      results: results.map((r) => ({
        citation: r.citation,
        url: r.url,
        speaker: r.speaker,
        year: r.date ? r.date.slice(0, 4) : "",
        kind: r.kind,
        title: r.title,
        text: r.displayText,
        window: r.windowText,
        rerank: r.rerank,
        dense: r.denseRank,
        lexical: r.lexicalRank,
      })),
    });
  } catch (err) {
    console.error("search failed", err);
    return NextResponse.json({ error: "search failed" }, { status: 500 });
  }
}
