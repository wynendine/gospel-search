"""General Conference talks, 1971-present, via the study API.

Talk slugs are not guessable — 2025 uses "11oaks", 1985 uses "born-of-god" —
so each conference's manifest is walked to discover the real URIs.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from ..config import CHURCH_HOST, FIRST_CONFERENCE_YEAR
from . import church_api

MONTHS = ("04", "10")
MONTH_NAMES = {"04": "April", "10": "October"}

# Bylines read "By President Russell M. Nelson" / "Presented by President
# Dallin H. Oaks" — strip the lead-in so the speaker field is just the name.
_BYLINE_LEAD = re.compile(r"^(?:By|Presented by|Read by|Delivered by)\s+", re.I)


@dataclass
class Talk:
    uri: str
    title: str
    speaker: str
    role: str
    year: int
    month: str
    paragraphs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"{CHURCH_HOST}/study{self.uri}?lang=eng"

    @property
    def date(self) -> str:
        return f"{self.year}-{self.month}-01"

    @property
    def conference(self) -> str:
        return f"{MONTH_NAMES[self.month]} {self.year}"


def conference_periods(through: dt.date | None = None) -> list[tuple[int, str]]:
    """Every (year, month) conference that has happened, oldest first."""
    today = through or dt.date.today()
    periods = []
    for year in range(FIRST_CONFERENCE_YEAR, today.year + 1):
        for month in MONTHS:
            if (year, int(month)) <= (today.year, today.month):
                periods.append((year, month))
    return periods


def talk_uris(year: int, month: str) -> list[str]:
    """Discover talk URIs by walking a conference manifest."""
    payload = church_api.fetch(f"/general-conference/{year}/{month}")
    if not payload:
        return []

    soup = BeautifulSoup(payload["content"]["body"], "lxml")
    prefix = f"/study/general-conference/{year}/{month}/"
    uris: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor["href"].split("?")[0]
        if not href.startswith(prefix):
            continue
        uri = href[len("/study") :]
        if uri not in uris:
            uris.append(uri)
    return uris


def _clean(soup_fragment) -> str:
    """Strip footnote markers and page breaks, then flatten to text."""
    for junk in soup_fragment.select("a.study-note-ref, sup.marker, span.page-break"):
        junk.decompose()
    return re.sub(r"\s+", " ", soup_fragment.get_text(" ", strip=True)).strip()


def parse_talk(uri: str, year: int, month: str) -> Talk | None:
    """Parse one talk. Returns None for session index pages and other non-talks."""
    payload = church_api.fetch(uri)
    if not payload:
        return None

    soup = BeautifulSoup(payload["content"]["body"], "lxml")

    byline = soup.select_one("div.byline")
    body = soup.select_one("div.body-block")
    if byline is None or body is None:
        return None  # session index / manifest page, not a talk

    paragraphs: list[tuple[str, str]] = []
    for node in body.select("p[data-aid]"):
        anchor = node.get("id")
        text = _clean(node)
        if anchor and len(text) > 1:
            paragraphs.append((anchor, text))

    if len(paragraphs) < 3:
        return None  # too thin to be a talk (stray notices, media-only pages)

    heading = soup.select_one("h1")
    speaker_node = byline.select_one("p.author-name")
    role_node = byline.select_one("p.author-role")

    return Talk(
        uri=uri,
        title=_clean(heading) if heading else payload["meta"].get("title", uri),
        speaker=_BYLINE_LEAD.sub("", _clean(speaker_node)) if speaker_node else "",
        role=_clean(role_node) if role_node else "",
        year=year,
        month=month,
        paragraphs=paragraphs,
    )


def iter_talks(periods=None, *, progress=None):
    """Yield every parseable talk across the given conference periods."""
    for year, month in periods or conference_periods():
        for uri in talk_uris(year, month):
            talk = parse_talk(uri, year, month)
            if progress is not None:
                progress(year, month, uri, talk is not None)
            if talk is not None:
                yield talk
