"""Turn talks and chapters into embeddable chunks.

Two rules do most of the work here:

1. Window for context, anchor on one unit. A lone paragraph or verse is too
   small to embed well; a whole talk is too big to be a precise hit. So the
   embedded text is a 3-unit sliding window, but the citation and the displayed
   passage are the *center* unit.

2. Prefix the embedded text with its provenance. Putting the speaker, title and
   date inside the vector is what makes "what did Nelson say about covenants"
   work at all — otherwise the speaker's name appears nowhere in the text being
   compared against the query.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import WINDOW


@dataclass
class Document:
    key: str  # stable unique id, e.g. the URI
    kind: str  # "talk" | "chapter"
    title: str
    url: str
    speaker: str = ""
    role: str = ""
    date: str = ""
    volume: str = ""
    book: str = ""
    chapter: int | None = None
    conference: str = ""


@dataclass
class Chunk:
    doc_key: str
    kind: str  # "talk" | "verse"
    anchor: str  # paragraph id or verse number — the deep-link target
    ordinal: int
    citation: str
    url: str
    display_text: str  # the center unit, shown in results
    window_text: str  # center plus neighbours, shown on expand
    embed_text: str  # what actually gets embedded


def quoted_title(title: str) -> str:
    """Wrap a talk title in quotes, unless it already is a quotation.

    Some titles are themselves quoted phrases and arrive with curly quotes
    already attached — “Lord, I Believe” — so wrapping unconditionally renders
    them as ""Lord, I Believe"".
    """
    if title[:1] in '"“' and title[-1:] in '"”':
        return title
    return f"“{title}”"


def _window(units: list, index: int, size: int = WINDOW) -> list:
    start = max(0, index - size)
    return units[start : index + size + 1]


def chunk_talk(talk) -> tuple[Document, list[Chunk]]:
    document = Document(
        key=talk.uri,
        kind="talk",
        title=talk.title,
        url=talk.url,
        speaker=talk.speaker,
        role=talk.role,
        date=talk.date,
        conference=talk.conference,
    )

    # The embedded header is deliberately left as-is: changing it would change
    # every chunk's embed_text and force a full re-embed of the talk corpus for
    # what is only a display concern. The citation is what readers see.
    header = f'{talk.speaker}, "{talk.title}" ({talk.conference} General Conference)'
    citation = f"{talk.speaker}, {quoted_title(talk.title)} ({talk.conference})"
    texts = [text for _, text in talk.paragraphs]
    chunks: list[Chunk] = []

    for index, (anchor, text) in enumerate(talk.paragraphs):
        window = _window(texts, index)
        window_text = "\n\n".join(window)
        chunks.append(
            Chunk(
                doc_key=talk.uri,
                kind="talk",
                anchor=anchor,
                ordinal=index,
                citation=citation,
                url=f"{talk.url}#{anchor}",
                display_text=text,
                window_text=window_text,
                embed_text=f"{header}\n\n{window_text}",
            )
        )

    return document, chunks


def chunk_chapter(chapter) -> tuple[Document, list[Chunk]]:
    document = Document(
        key=chapter.uri,
        kind="chapter",
        title=chapter.citation,
        url=chapter.url,
        volume=chapter.volume,
        book=chapter.book,
        chapter=chapter.number,
    )

    header = f"{chapter.citation} ({chapter.volume})"
    texts = [verse.text for verse in chapter.verses]
    chunks: list[Chunk] = []

    for index, verse in enumerate(chapter.verses):
        window_text = " ".join(_window(texts, index))
        chunks.append(
            Chunk(
                doc_key=chapter.uri,
                kind="verse",
                anchor=str(verse.number),
                ordinal=index,
                citation=verse.citation,
                url=verse.url,
                display_text=verse.text,
                window_text=window_text,
                embed_text=f"{header}\n\n{window_text}",
            )
        )


    return document, chunks
