"""Claude synthesis over retrieved passages.

The answer is a convenience; the passages are the product. Every claim carries
a [n] pointing at a numbered passage, and the model is told plainly to say when
the passages don't support an answer rather than reaching past them.
"""

from __future__ import annotations

from .config import ANSWER_MODEL
from .search import claude

PROMPT = """You are helping someone find where something was said in General \
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
- Quote sparingly and only from the passages given."""


def format_passages(results) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        header = result.citation
        if result.kind == "talk" and result.date:
            header = f"{header} — {result.date[:4]}"
        blocks.append(f"[{i}] {header}\n{result.window_text}")
    return "\n\n".join(blocks)


def answer(query: str, results, *, model: str = ANSWER_MODEL) -> str:
    if not results:
        return "Nothing in the index matched that query."

    response = claude().messages.create(
        model=model,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    query=query, passages=format_passages(results)
                ),
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
