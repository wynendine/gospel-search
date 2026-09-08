"""Thin, polite, on-disk-cached client for the Church's study API.

Every page we need — conference manifests, talks, scripture chapters — comes
from one endpoint that returns `{meta, content: {head, body, footnotes}}` where
`body` is clean, machine-generated HTML. Responses are cached under data/raw so
a re-run only fetches what's missing and an interrupted crawl resumes for free.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import FETCH_WORKERS, RAW, REQUEST_DELAY, STUDY_API, USER_AGENT

_throttle = threading.Lock()
_last_request = 0.0


def _wait_turn() -> None:
    """Gate every worker through one clock, for a flat global request rate.

    Sleeping while holding the lock is deliberate: it serializes only the
    *scheduling*, so the rate stays at 1/REQUEST_DELAY no matter how many
    workers there are, while the requests themselves still overlap.
    """
    global _last_request
    with _throttle:
        elapsed = time.monotonic() - _last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        _last_request = time.monotonic()


def _cache_path(uri: str) -> Path:
    digest = hashlib.sha1(uri.encode()).hexdigest()[:16]
    slug = uri.strip("/").replace("/", "_")[:80]
    return RAW / "study" / f"{slug}.{digest}.json"


def fetch(uri: str, *, refresh: bool = False) -> dict | None:
    """Return the study-API payload for `uri`, or None if it 404s.

    `uri` is the site-relative path with no /study prefix, e.g.
    "/general-conference/2025/04/11oaks" or "/scriptures/bofm/1-ne/3".
    """
    path = _cache_path(uri)
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink()  # truncated write from an interrupted run; refetch

    url = f"{STUDY_API}?{urllib.parse.urlencode({'lang': 'eng', 'uri': uri})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    _wait_turn()

    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 3:
                time.sleep(2**attempt)
                continue
            raise
    else:  # pragma: no cover - loop always breaks or raises
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic, so a kill mid-write can't poison the cache
    return payload


def prefetch(uris, *, workers: int = FETCH_WORKERS, progress=None) -> int:
    """Warm the cache for many URIs at once.

    Parsing is cheap and fetching is not, so the crawl is split in two: pull
    everything concurrently here, then walk the cache serially. Failures are
    swallowed — a URI that won't load is simply absent from the cache, and the
    parse pass skips it.
    """
    todo = [uri for uri in uris if not _cache_path(uri).exists()]
    if not todo:
        if progress is not None:
            progress(0, 0)
        return 0

    done = 0

    def one(uri):
        try:
            fetch(uri)
        except Exception:  # noqa: BLE001 - a dead URI must not kill the crawl
            return

    # as_completed, not map: map yields in submission order, so one slow page
    # stalls the counter while dozens finish behind it — progress that reads as
    # a stall when the crawl is actually at full speed.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, uri) for uri in todo]
        for _ in as_completed(futures):
            done += 1
            if progress is not None:
                progress(done, len(todo))

    return done
