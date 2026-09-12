"""Claude synthesis over retrieved passages.

The answer is a convenience; the passages are the product. Every claim carries
a [n] pointing at a numbered passage, and the model is told plainly to say when
the passages don't support an answer rather than reaching past them.

Each intent gets its own instructions. "Lead with the most likely match" is
right for a lookup and actively wrong for a comparison, which needs both sides
weighted, or an enumeration, which needs a list rather than an argument.
"""

from __future__ import annotations

from . import config
from .config import ANSWER_MODEL
from . import usage
from .search import claude

BASE = """You are helping someone study General Conference talks and the \
standard works.

Their question: {query}

Numbered passages retrieved for it:

{passages}
"""

HONESTY = """
- Cite every claim with the passage number in brackets, like [3].
- Use only the passages given. If they do not support an answer, say so plainly \
in one sentence rather than filling the gap from your own knowledge — an honest \
miss is more useful than a confident wrong reference.
- Quote sparingly and only from the passages."""

INSTRUCTIONS = {
    "lookup": """Write a short answer — usually 2-4 sentences.

- Lead with the most likely match. If one passage is clearly what they meant, \
say so and name the speaker and talk, or the scripture reference.
- If several passages say similar things, point at the best one rather than \
listing them all; they can see the full list below.""",
    "enumerate": """List what was found.

- Group by book or speaker, in the order given — the passages are already in \
canonical order, so do not re-rank them.
- Give the reference for each with a few words on what it says. Do not write an \
essay; this is an index, not an argument.
- State the count plainly.
- If the results are complete, say so. If they are not, say that instead — \
never imply completeness you were not given.
- When you are given a distribution instead of a full list, lead with the total \
and where the matches concentrate, then give the examples as examples. Do not \
present a sample as though it were the index, and say what narrowing would \
produce a complete answer.""",
    "compare": """Compare what each person said.

- Give each person their own treatment before you compare them. Do not let the \
one with more passages dominate.
- Say what they share, then where they differ — in emphasis, framing, or the \
situation each is addressing. "They both testify of Christ" is true of everyone \
and worth nothing; find the real distinction.
- If the passages for someone are thin or from a single talk, say so. A \
comparison built on one talk each is not a survey of their teaching, and the \
reader needs to know which they are getting.
- Anchor every characterization in a quoted phrase with its citation.""",
    "thematic": """Summarize the teaching on this subject.

- The passages arrive grouped by the angle that surfaced them. Treat those as a
  starting structure, not a mandate: merge angles that turned out to say the
  same thing, and split one that clearly holds two ideas.
- Organize by idea, never by speaker or by rank.
- Prefer breadth. Drawing on many sources is the point of this shape of answer,
  so do not spend the whole response on the two strongest passages.
- Note where sources agree, and name any real difference in emphasis. "They all
  testify of Christ" is true of everything here and worth nothing.
- Where the teaching developed over time, say so and date it.
- Be clear this is what the retrieved passages show, not the whole of what has
  been taught.""",
}


def format_passages(results, groups=None) -> str:
    """Number passages for citation, grouped when the intent has groups.

    Comparisons group by person, thematic surveys by facet. Numbering runs
    across the whole set so a citation means the same thing either way.
    """
    if groups:
        blocks, index = [], 1
        for label, items in groups.items():
            if not items:
                continue
            blocks.append(f"--- {label} ---")
            for r in items:
                blocks.append(f"[{index}] {r.citation}\n{r.window_text}")
                index += 1
        return "\n\n".join(blocks)

    return "\n\n".join(
        f"[{i}] {r.citation}"
        + (f" — {r.date[:4]}" if r.kind == "talk" and r.date else "")
        + f"\n{r.window_text}"
        for i, r in enumerate(results, start=1)
    )


def answer(query: str, findings, *, model: str | None = None) -> str:
    if not findings.results:
        return "Nothing in the index matched that query."

    model = model or config.ANSWER_MODEL
    intent = findings.plan.intent
    prompt = (
        BASE.format(
            query=query,
            passages=format_passages(findings.results, findings.groups),
        )
        + "\n"
        + INSTRUCTIONS.get(intent, INSTRUCTIONS["lookup"])
        + HONESTY
    )

    if findings.exhaustive and findings.total_matches is not None:
        prompt += (
            f"\n\nThese are ALL {findings.total_matches} matches in the corpus "
            "for this term under the stated scope — the list is complete."
        )
    if findings.breakdown:
        rows = "\n".join(f"  {g}: {n:,}" for g, n in findings.breakdown)
        prompt += f"\n\nDistribution of all {findings.total_matches:,} matches — {findings.breakdown_label}:\n{rows}"
    if findings.note:
        prompt += f"\n\nImportant caveat you must convey: {findings.note}"

    usage.check_budget()
    response = claude().messages.create(
        model=model,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    usage.record_response("answer", model, response)
    return "".join(b.text for b in response.content if b.type == "text").strip()
