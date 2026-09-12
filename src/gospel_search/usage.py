"""Record what each call actually cost, so spend is observed rather than feared.

Per-token pricing makes the bill invisible until it arrives. Every Claude call
logs its real token counts here, priced from a table, so `gospel spend` answers
"what am I actually spending" from measurement instead of estimate — and a
monthly ceiling can stop the tool rather than surprise you.

Prices are per million tokens, from Anthropic's published rates.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
import time
from datetime import datetime, timezone

from .config import INDEX

USAGE_PATH = INDEX / "usage.db"

PRICES = {  # model prefix -> (input $/MTok, output $/MTok)
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}
EMBED_PRICES = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}

# Set GOSPEL_MONTHLY_LIMIT to a dollar figure to have the tool refuse to spend
# past it. Unset means no ceiling.
MONTHLY_LIMIT = float(os.environ.get("GOSPEL_MONTHLY_LIMIT", "0") or 0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id      INTEGER PRIMARY KEY,
    at      REAL NOT NULL,
    month   TEXT NOT NULL,
    run     TEXT,
    stage   TEXT NOT NULL,
    model   TEXT NOT NULL,
    tok_in  INTEGER NOT NULL,
    tok_out INTEGER NOT NULL,
    cost    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_month ON calls(month);
CREATE INDEX IF NOT EXISTS calls_run   ON calls(run);
"""


# Calls are grouped into a "run" — one thing the user asked for — so cost per
# search is measured rather than inferred by dividing totals by a stage count,
# which over-attributes as soon as a --no-answer search is in the mix.
_current_run: str | None = None


def start_run() -> str:
    global _current_run
    _current_run = uuid.uuid4().hex[:12]
    return _current_run


def end_run() -> float:
    """Dollars spent on the run just finished."""
    global _current_run
    if _current_run is None:
        return 0.0
    conn = _connect()
    total = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) FROM calls WHERE run = ?", (_current_run,)
    ).fetchone()[0]
    conn.close()
    _current_run = None
    return total


class BudgetExceeded(RuntimeError):
    """Raised instead of spending past GOSPEL_MONTHLY_LIMIT."""


def _connect() -> sqlite3.Connection:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USAGE_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def price(model: str, tok_in: int, tok_out: int) -> float:
    for prefix, (pin, pout) in PRICES.items():
        if model.startswith(prefix):
            return tok_in * pin / 1e6 + tok_out * pout / 1e6
    return 0.0  # unknown model: record it, don't guess a number


def this_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def spent_this_month() -> float:
    conn = _connect()
    total = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) FROM calls WHERE month = ?", (this_month(),)
    ).fetchone()[0]
    conn.close()
    return total


def check_budget() -> None:
    """Refuse to start a call that would run past the monthly ceiling."""
    if MONTHLY_LIMIT <= 0:
        return
    spent = spent_this_month()
    if spent >= MONTHLY_LIMIT:
        raise BudgetExceeded(
            f"${spent:.2f} spent this month, limit is ${MONTHLY_LIMIT:.2f}. "
            "Raise GOSPEL_MONTHLY_LIMIT or wait for the month to roll over. "
            "References, `gospel cites` and cached queries still work — they cost nothing."
        )


def record(stage: str, model: str, tok_in: int, tok_out: int) -> float:
    cost = price(model, tok_in, tok_out)
    conn = _connect()
    conn.execute(
        """INSERT INTO calls (at, month, run, stage, model, tok_in, tok_out, cost)
           VALUES (?,?,?,?,?,?,?,?)""",
        (time.time(), this_month(), _current_run, stage, model, tok_in, tok_out, cost),
    )
    conn.commit()
    conn.close()
    return cost


def record_response(stage: str, model: str, response) -> float:
    """Log a Claude response's real usage — never an estimate."""
    try:
        return record(stage, model, response.usage.input_tokens, response.usage.output_tokens)
    except Exception:  # noqa: BLE001 - accounting must never break a search
        return 0.0


def record_embedding(model: str, tokens: int) -> float:
    cost = tokens * EMBED_PRICES.get(model, 0.0) / 1e6
    conn = _connect()
    conn.execute(
        """INSERT INTO calls (at, month, run, stage, model, tok_in, tok_out, cost)
           VALUES (?,?,?,?,?,?,?,?)""",
        (time.time(), this_month(), _current_run, "embed", model, tokens, 0, cost),
    )
    conn.commit()
    conn.close()
    return cost


def summary(months: int = 6) -> dict:
    conn = _connect()
    by_month = conn.execute(
        """SELECT month, COUNT(*) AS calls, SUM(cost) AS cost FROM calls
           GROUP BY month ORDER BY month DESC LIMIT ?""",
        (months,),
    ).fetchall()
    by_stage = conn.execute(
        """SELECT stage, COUNT(*) AS calls, SUM(cost) AS cost FROM calls
           WHERE month = ? GROUP BY stage ORDER BY cost DESC""",
        (this_month(),),
    ).fetchall()
    runs = conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(AVG(c), 0) AS avg FROM (
             SELECT SUM(cost) AS c FROM calls
              WHERE month = ? AND run IS NOT NULL GROUP BY run)""",
        (this_month(),),
    ).fetchone()
    conn.close()
    return {
        "by_month": [dict(r) for r in by_month],
        "by_stage": [dict(r) for r in by_stage],
        "searches": runs["n"],
        "per_search": runs["avg"],
        "limit": MONTHLY_LIMIT,
        "spent": spent_this_month(),
    }
