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
# Each stage is set independently, because they are not the same kind of work.
# Synthesis is the one that has to reason over a dozen passages and write
# something worth reading; planning, query expansion and reranking are narrow
# judgment tasks. Override any of them per-run with an env var, which is how
# the cost/quality numbers in the README were measured:
#
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
