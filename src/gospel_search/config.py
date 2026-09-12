"""Paths, model names, and tunables. Everything configurable lives here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index"

DB_PATH = INDEX / "corpus.db"
VECTORS_PATH = INDEX / "vectors.npy"
META_PATH = INDEX / "meta.json"

# --- Embeddings ------------------------------------------------------------
# 3-large truncated to 1024 dims (matryoshka): keeps nearly all the quality and
# cuts the vector store from 2.3 GB to ~790 MB at full corpus size.
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1024
EMBED_BATCH = 256

# --- Claude ----------------------------------------------------------------
# Every stage runs on Opus 5. Cheaper models were measured at each stage rather
# than assumed, and none of them held up:
#
#   HyDE on Haiku       hit@1 91% -> 77%   it writes a passage in scriptural
#                                          register, which is generation work
#   planning on Haiku   hit@1 91% -> 86%
#   rerank on Haiku     0.941, 0.924, 0.827 MRR across three runs
#
# The rerank result is the cautionary one. A single run came back at 0.941 —
# inside the Opus spread — and that one sample was very nearly taken as proof
# it was free. Two more runs showed the real problem is variance: Haiku's
# reranking is not slightly worse, it is *inconsistent*, and a search tool that
# is excellent two times in three is worse than one that is merely very good
# every time. At a few dollars a month the saving does not buy anything worth
# having.
#
# Override per run if you want to re-measure:
#   GOSPEL_RERANK_MODEL=claude-haiku-4-5 gospel eval
ANSWER_MODEL = os.environ.get("GOSPEL_ANSWER_MODEL", "claude-opus-5")
RERANK_MODEL = os.environ.get("GOSPEL_RERANK_MODEL", "claude-opus-5")
HYDE_MODEL = os.environ.get("GOSPEL_HYDE_MODEL", "claude-opus-5")

# `effort` arrived with the 4.6 generation. Older models reject it outright, so
# a stage pointed at one has to omit the parameter rather than send a default.
NO_EFFORT_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-3")


def output_config(model: str, **fields) -> dict:
    """output_config for a model, dropping `effort` where it isn't supported."""
    if any(model.startswith(m) for m in NO_EFFORT_MODELS):
        fields.pop("effort", None)
    return fields


# --- Retrieval -------------------------------------------------------------
DENSE_K = 100  # candidates from the vector half
LEXICAL_K = 100  # candidates from the BM25 half
RRF_K = 60  # reciprocal-rank-fusion damping constant
RERANK_K = int(os.environ.get("GOSPEL_RERANK_K", "50"))  # candidates reranked
RERANK_CHARS = int(os.environ.get("GOSPEL_RERANK_CHARS", "700"))
ANSWER_K = 12  # reranked passages handed to the synthesizer

# Cap on passages from any one talk or chapter. Ranking is per-chunk, so a
# single strongly on-topic talk otherwise wins most of the slots — a two-speaker
# comparison drew six of eight passages from two talks before this existed.
# Raise it when you want depth in one source, lower it for breadth.
MAX_PER_DOC = 3

# Query planning
PLAN_MODEL = os.environ.get("GOSPEL_PLAN_MODEL", "claude-opus-5")
COMPARE_BUDGET = 8   # passages retrieved per entity in a comparison
ENUMERATE_CAP = 200  # hard ceiling on an exhaustive listing
THEMATIC_BUDGET = 6  # passages retrieved per facet of a thematic survey

WINDOW = 1  # units of context on each side of the anchor (1 => 3-unit window)

# --- Sources ---------------------------------------------------------------
CHURCH_HOST = "https://www.churchofjesuschrist.org"
STUDY_API = f"{CHURCH_HOST}/study/api/v3/language-pages/type/content"
SCRIPTURES_CSV_URL = (
    "https://raw.githubusercontent.com/beandog/lds-scriptures/master/csv/lds-scriptures.csv"
)
FIRST_CONFERENCE_YEAR = 1971
USER_AGENT = "gospel-search/0.1 (personal study tool; contact: local user)"

# Each study-API request takes ~0.5 s, almost all of it waiting on the network,
# so a serial crawl of ~4,000 pages would run for hours doing nothing. A small
# pool with a per-worker delay lands around 5 requests/second — roughly what a
# browser does opening one page, and it finishes in ~15 minutes.
FETCH_WORKERS = 5
REQUEST_DELAY = 0.25  # seconds a worker waits before its next request


def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in gospel-search/.env)")
    return key


def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (put it in gospel-search/.env)")
    return key
