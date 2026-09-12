"""Read a question and decide how to search for it.

Everything used to route through one strategy: embed the query, take top-k,
synthesize. That is right for "find the talk I half-remember" and wrong for
"every reference to X in the Book of Mormon" (which needs completeness, not
ranking) and for "compare Bednar and Holland on faith" (which needs one
retrieval per person — a single blended query vector has no idea the question
names two people, and returns whoever happens to rank well).

The planner is one small Claude call that classifies the question and pulls the
scoping out of the prose, so `--volume "Book of Mormon"` no longer has to be
passed by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config
from .config import PLAN_MODEL
from . import usage
from .search import Filters, claude

VOLUMES = [
    "Old Testament",
    "New Testament",
    "Book of Mormon",
    "Doctrine and Covenants",
    "Pearl of Great Price",
]

PROMPT = """Classify a search over General Conference talks (April 1971 - April \
2026) and the standard works, and pull any scoping out of the wording.

Intents:
  lookup    — find a specific half-remembered passage or talk.
              "the talk about the father whose son was ill"
  enumerate — list occurrences, aiming for completeness rather than the best few.
              "every reference to charity in the Book of Mormon"
  compare   — contrast what two or more people (or sources) said about a topic.
              "compare Bednar and Holland on faith"
  thematic  — survey teaching on a subject broadly across many sources.
              "what has been taught about the gathering of Israel"
  citations — which talks quote or cite a specific scripture.
              "which talks cite Alma 32:21", "who has quoted Moses 1:39"

Rules:
- `topic` is what to search for, stripped of scoping and of the instruction
  itself. For "every reference to charity in the Book of Mormon" the topic is
  "charity", not the whole sentence.
- `literal_terms` matters only for enumerate: the exact words to match. Set it
  when the user is asking for occurrences of a *word or phrase* ("references to
  charity"). Leave it empty when they mean a *concept* with no fixed wording
  ("every passage about enduring trials") — that cannot be enumerated exactly,
  and the caller needs to know the difference.
- For citations, put the scripture reference in `topic` exactly as given
  ("Alma 32:21"), and nothing else.
- `facets` matters only for thematic: 4-6 distinct sub-questions that together
  cover the subject, each phrased as something you would search for on its own.
  Make them genuinely different angles, not restatements — for "the gathering
  of Israel" that means temple and family history work, missionary work, the
  scattering itself, covenant and adoption, and who is asked to participate.
  One broad query finds one facet and misses the rest.
- `match_mode` matters only for enumerate. "all" when the question asks for
  passages containing every term together ("faith and repentance together"),
  "any" when the terms are alternatives or spellings of one idea ("references
  to charity"). Default to "any".
- `entities` matters only for compare: the speakers or sources to contrast,
  as surnames for people ("Bednar", "Holland").
- Volumes must be exactly one of: {volumes}. Books are single books like
  "Alma", "Isaiah", "Moroni".
- source is "scriptures" when the question is about the standard works,
  "talks" when it is about conference talks, null when either could answer.
- Years: after/before are conference years, only for talk questions.
- Set nothing you are not told. Do not infer a speaker from a topic.

Question: {query}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["lookup", "enumerate", "compare", "thematic", "citations"]},
        "topic": {"type": "string"},
        "literal_terms": {"type": "array", "items": {"type": "string"}},
        "match_mode": {"type": "string", "enum": ["any", "all"]},
        "entities": {"type": "array", "items": {"type": "string"}},
        "facets": {"type": "array", "items": {"type": "string"}},
        # A nullable field cannot also carry an enum here, so the allowed
        # values are enforced below rather than by the schema.
        "source": {"type": ["string", "null"]},
        "volume": {"type": ["string", "null"]},
        "book": {"type": ["string", "null"]},
        "speaker": {"type": ["string", "null"]},
        "after": {"type": ["integer", "null"]},
        "before": {"type": ["integer", "null"]},
    },
    "required": [
        "intent", "topic", "literal_terms", "match_mode", "entities", "facets",
        "source", "volume", "book", "speaker", "after", "before",
    ],
    "additionalProperties": False,
}


@dataclass
class Plan:
    intent: str = "lookup"
    topic: str = ""
    literal_terms: list[str] = field(default_factory=list)
    match_mode: str = "any"
    entities: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)

    def describe(self) -> str:
        bits = [f"intent={self.intent}", f'topic="{self.topic}"']
        if self.literal_terms:
            bits.append(f"terms={self.literal_terms} ({self.match_mode})")
        if self.entities:
            bits.append(f"entities={self.entities}")
        if self.facets:
            bits.append(f"facets={len(self.facets)}")
        for name in ("source", "volume", "book", "speaker", "after", "before"):
            value = getattr(self.filters, name)
            if value:
                bits.append(f"{name}={value}")
        return " · ".join(bits)


def plan(query: str, *, override: Filters | None = None) -> Plan:
    """Classify a query. Falls back to a plain lookup if the call fails."""
    try:
        usage.check_budget()
        response = claude().messages.create(
            model=config.PLAN_MODEL,
            max_tokens=800,
            output_config=config.output_config(
                config.PLAN_MODEL,
                effort="low",
                format={"type": "json_schema", "schema": SCHEMA},
            ),
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        query=query, volumes=", ".join(VOLUMES)
                    ),
                }
            ],
        )
        usage.record_response("plan", config.PLAN_MODEL, response)
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001 - planning degrades, it doesn't block
        # Never fail silently here: a plan that quietly becomes a plain lookup
        # turns "every reference to X" into "the ten best passages about X"
        # with nothing in the output admitting the difference.
        print(f"  (planning failed: {type(exc).__name__}; treating as a lookup)")
        return Plan(topic=query)

    source = data.get("source")
    if source not in ("talks", "scriptures"):
        source = None

    # Match a loosely-worded volume ("BoM", "the Book of Mormon") to the exact
    # string the corpus uses, and drop it if it matches nothing.
    volume = data.get("volume")
    if volume:
        volume = next(
            (v for v in VOLUMES if v.lower() in volume.lower() or volume.lower() in v.lower()),
            None,
        )

    filters = Filters(
        speaker=data.get("speaker"),
        after=data.get("after"),
        before=data.get("before"),
        source=source,
        volume=volume,
        book=data.get("book"),
    )

    # Explicit CLI flags beat anything inferred from the prose.
    if override is not None:
        for name in ("speaker", "after", "before", "source", "volume", "book"):
            value = getattr(override, name)
            if value:
                setattr(filters, name, value)

    return Plan(
        intent=data.get("intent", "lookup"),
        topic=data.get("topic") or query,
        literal_terms=[t for t in data.get("literal_terms", []) if t],
        match_mode="all" if data.get("match_mode") == "all" else "any",
        entities=[e for e in data.get("entities", []) if e],
        facets=[f for f in data.get("facets", []) if f],
        filters=filters,
    )
