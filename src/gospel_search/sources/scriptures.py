"""The standard works, from the public-domain lds-scriptures CSV.

One 13.7 MB download gives all 41,995 verses across 87 books, with citations
("Alma 32:21") already formatted and URL slugs that match the live site.
Chapter study summaries are not in the CSV — those come from the study API as
an optional second pass.
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from ..config import CHURCH_HOST, RAW, SCRIPTURES_CSV_URL, USER_AGENT
from . import church_api

CSV_PATH = RAW / "lds-scriptures.csv"


@dataclass
class Verse:
    citation: str  # "Alma 32:21"
    short_citation: str  # "Alma 32:21" / "Gen. 1:1"
    text: str
    volume: str
    book: str
    chapter: int
    number: int
    volume_slug: str
    book_slug: str

    @property
    def chapter_uri(self) -> str:
        return f"/scriptures/{self.volume_slug}/{self.book_slug}/{self.chapter}"

    @property
    def url(self) -> str:
        return f"{CHURCH_HOST}/study{self.chapter_uri}?lang=eng#p{self.number}"


@dataclass
class Chapter:
    volume: str
    book: str
    number: int
    volume_slug: str
    book_slug: str
    verses: list[Verse] = field(default_factory=list)
    summary: str = ""

    @property
    def citation(self) -> str:
        return f"{self.book} {self.number}"

    @property
    def uri(self) -> str:
        return f"/scriptures/{self.volume_slug}/{self.book_slug}/{self.number}"

    @property
    def url(self) -> str:
        return f"{CHURCH_HOST}/study{self.uri}?lang=eng"


def download_csv(path: Path = CSV_PATH, *, refresh: bool = False) -> Path:
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SCRIPTURES_CSV_URL, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(response.read())
    tmp.replace(path)
    return path


def iter_chapters(path: Path = CSV_PATH):
    """Yield chapters in canonical order, each with its verses attached."""
    download_csv(path)
    current: Chapter | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            verse = Verse(
                citation=row["verse_title"],
                short_citation=row["verse_short_title"],
                text=row["scripture_text"],
                volume=row["volume_title"],
                book=row["book_title"],
                chapter=int(row["chapter_number"]),
                number=int(row["verse_number"]),
                volume_slug=row["volume_lds_url"],
                book_slug=row["book_lds_url"],
            )
            key = (verse.book_slug, verse.chapter)
            if current is None or (current.book_slug, current.number) != key:
                if current is not None:
                    yield current
                current = Chapter(
                    volume=verse.volume,
                    book=verse.book,
                    number=verse.chapter,
                    volume_slug=verse.volume_slug,
                    book_slug=verse.book_slug,
                )
            current.verses.append(verse)

    if current is not None:
        yield current


# Most chapters carry a topical head-note, but a few don't and use something
# else instead: short D&C sections have only the historical `study-intro`, and
# Joseph Smith—History has only a `subtitle`. Ordered best-first; chapters that
# have a real summary never reach the fallbacks.
SUMMARY_SELECTORS = ("p.study-summary", "p.study-intro", "p.subtitle")


def fetch_summary(chapter: Chapter) -> str:
    """Chapter head-note from the study API ('Lehi's sons return to Jerusalem...')."""
    payload = church_api.fetch(chapter.uri)
    if not payload:
        return ""
    soup = BeautifulSoup(payload["content"]["body"], "lxml")
    for selector in SUMMARY_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return ""  # Psalms 119 and Articles of Faith genuinely have no head-note


def verify_slugs(chapters) -> list[str]:
    """Confirm one URL per volume actually resolves. Returns failures."""
    seen: set[str] = set()
    failures: list[str] = []
    for chapter in chapters:
        if chapter.volume_slug in seen:
            continue
        seen.add(chapter.volume_slug)
        if church_api.fetch(chapter.uri) is None:
            failures.append(f"{chapter.volume} -> {chapter.uri}")
    return failures
