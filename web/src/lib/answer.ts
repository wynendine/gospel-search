import Anthropic from "@anthropic-ai/sdk";

import { ANSWER_K, ANSWER_MODEL, type Result } from "@/lib/search";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const PROMPT = `You are helping someone find where something was said in General \
Conference talks and the standard works.

Their query: {query}

Numbered passages retrieved for that query:

{passages}

Write a short answer — usually 2-4 sentences, at most a paragraph.

- Cite every claim with the passage number in brackets, like [3].
- Lead with the most likely match. If one passage is clearly what they meant, \
say so and name the speaker and talk, or the scripture reference.
- If several passages say similar things, point at the best one rather than \
listing them all; they can see the full list below your answer.
- If the passages do not actually contain what they are looking for, say that \
plainly in one sentence. Do not fill the gap from your own knowledge — an \
honest miss is more useful than a confident wrong reference.
- Quote sparingly and only from the passages given.`;

export function formatPassages(results: Result[]): string {
  return results
    .map((r, i) => {
      const year = r.kind === "talk" && r.date ? ` — ${r.date.slice(0, 4)}` : "";
      return `[${i + 1}] ${r.citation}${year}\n${r.windowText}`;
    })
    .join("\n\n");
}

export async function answer(q: string, results: Result[]): Promise<string> {
  if (!results.length) return "Nothing in the index matched that query.";

  const res = await anthropic.messages.create({
    model: ANSWER_MODEL,
    max_tokens: 1500,
    messages: [
      {
        role: "user",
        content: PROMPT.replace("{query}", q).replace(
          "{passages}",
          formatPassages(results.slice(0, ANSWER_K)),
        ),
      },
    ],
  });

  return res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("")
    .trim();
}
