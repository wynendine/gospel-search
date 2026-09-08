"""Local web UI — one page, served from localhost."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import answer as answer_mod
from . import index as idx
from . import search as search_mod

app = FastAPI(title="Gospel Search")

_state: dict = {}


def _resources():
    if "conn" not in _state:
        _state["conn"] = idx.connect(readonly=True)
        _state["vectors"] = idx.load_vectors()
    return _state["conn"], _state["vectors"]


class Query(BaseModel):
    query: str
    n: int = 10
    speaker: str | None = None
    after: int | None = None
    before: int | None = None
    source: str | None = None
    answer: bool = True
    hyde: bool = True
    rerank: bool = True


@app.post("/api/search")
def api_search(request: Query):
    conn, vectors = _resources()
    results = search_mod.search(
        request.query,
        n=request.n,
        filters=search_mod.Filters(
            speaker=request.speaker,
            after=request.after,
            before=request.before,
            source=request.source,
        ),
        use_hyde=request.hyde,
        use_rerank=request.rerank,
        conn=conn,
        vectors=vectors,
    )

    text = ""
    if request.answer and results:
        text = answer_mod.answer(
            request.query, search_mod.context_for_answer(results)
        )

    return {
        "answer": text,
        "results": [
            {
                "citation": r.citation,
                "url": r.url,
                "speaker": r.speaker,
                "year": r.date[:4] if r.date else "",
                "kind": r.kind,
                "title": r.title,
                "text": r.display_text,
                "window": r.window_text,
                "rerank": r.rerank,
                "dense": r.dense_rank,
                "lexical": r.lexical_rank,
            }
            for r in results
        ],
    }


@app.get("/api/stats")
def api_stats():
    return idx.read_meta()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gospel Search</title>
<style>
  :root {
    --bg: #fbfaf8; --panel: #ffffff; --ink: #1b1a17; --muted: #6b6862;
    --line: #e5e1da; --accent: #7a5c2e; --on-accent: #ffffff;
    --accent-soft: #f3ede2;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161513; --panel: #1e1d1a; --ink: #eceae5; --muted: #97928a;
      --line: #302e2a; --accent: #d3ac6a; --on-accent: #1b1a17;
      --accent-soft: #262320;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 ui-serif, Georgia, "Times New Roman", serif;
  }
  header { padding: 28px 20px 8px; max-width: 860px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 2px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; margin: 0 0 18px; }
  main { max-width: 860px; margin: 0 auto; padding: 0 20px 80px; }
  form { display: flex; gap: 8px; margin-bottom: 10px; }
  input[type=text] {
    flex: 1; padding: 12px 14px; font: inherit; color: var(--ink);
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    padding: 12px 20px; font: inherit; font-weight: 600; cursor: pointer;
    background: var(--accent); color: var(--on-accent); border: 0; border-radius: 8px;
  }
  button:disabled { opacity: .55; cursor: default; }
  .filters {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
    color: var(--muted); margin-bottom: 22px;
  }
  .filters input, .filters select {
    font: inherit; padding: 5px 8px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  }
  .filters input[type=text] { width: 130px; }
  .filters input[type=number] { width: 78px; }
  label.check { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
  #answerBox {
    background: var(--accent-soft); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px 18px; margin-bottom: 26px;
    font-size: 15.5px; white-space: pre-wrap;
  }
  .card { border-top: 1px solid var(--line); padding: 18px 0; }
  .cite {
    font-family: ui-sans-serif, system-ui, sans-serif; font-weight: 600;
    font-size: 14px; color: var(--accent); text-decoration: none;
  }
  .cite:hover { text-decoration: underline; }
  .meta {
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    color: var(--muted); margin: 2px 0 8px;
  }
  .passage { margin: 0; }
  .more {
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    color: var(--muted); background: none; border: 0; padding: 6px 0 0;
    cursor: pointer; text-decoration: underline;
  }
  .window {
    margin-top: 8px; padding-left: 14px; border-left: 2px solid var(--line);
    color: var(--muted);
  }
  .status { color: var(--muted); font-size: 14px; padding: 20px 0; }
</style>
</head>
<body>
<header>
  <h1>Gospel Search</h1>
  <p class="sub">General Conference talks and the standard works &mdash; searched by meaning, not keyword.</p>
</header>
<main>
  <form id="f">
    <input type="text" id="q" placeholder="Describe what was said&hellip;" autofocus autocomplete="off">
    <button type="submit" id="go">Search</button>
  </form>

  <div class="filters">
    <select id="source">
      <option value="">everything</option>
      <option value="talks">talks only</option>
      <option value="scriptures">scriptures only</option>
    </select>
    <input type="text" id="speaker" placeholder="speaker">
    <input type="number" id="after" placeholder="after" min="1971" max="2100">
    <input type="number" id="before" placeholder="before" min="1971" max="2100">
    <label class="check"><input type="checkbox" id="answer" checked> answer</label>
    <label class="check"><input type="checkbox" id="rerank" checked> rerank</label>
    <label class="check"><input type="checkbox" id="hyde" checked> hyde</label>
  </div>

  <div id="answerBox" hidden></div>
  <div id="out"></div>
</main>

<script>
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = el => el.value ? parseInt(el.value, 10) : null;

$('f').addEventListener('submit', async e => {
  e.preventDefault();
  const query = $('q').value.trim();
  if (!query) return;

  $('go').disabled = true;
  $('answerBox').hidden = true;
  $('out').innerHTML = '<p class="status">Searching&hellip;</p>';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        query, n: 10,
        speaker: $('speaker').value.trim() || null,
        after: num($('after')), before: num($('before')),
        source: $('source').value || null,
        answer: $('answer').checked,
        rerank: $('rerank').checked,
        hyde: $('hyde').checked,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    if (data.answer) {
      $('answerBox').textContent = data.answer;
      $('answerBox').hidden = false;
    }

    if (!data.results.length) {
      $('out').innerHTML = '<p class="status">No matches.</p>';
    } else {
      $('out').innerHTML = data.results.map((r, i) => {
        const bits = [];
        if (r.speaker) bits.push(esc(r.speaker));
        if (r.year) bits.push(r.year);
        if (r.rerank !== null) bits.push(`rerank ${Math.round(r.rerank)}/10`);
        const halves = [r.dense && 'dense', r.lexical && 'bm25'].filter(Boolean);
        if (halves.length) bits.push(halves.join('+'));
        return `<div class="card">
          <a class="cite" href="${esc(r.url)}" target="_blank" rel="noopener">${i+1}. ${esc(r.citation)}</a>
          <div class="meta">${bits.join(' &middot; ')}</div>
          <p class="passage">${esc(r.text)}</p>
          <button class="more" data-i="${i}">show surrounding</button>
          <div class="window" id="w${i}" hidden>${esc(r.window)}</div>
        </div>`;
      }).join('');

      $('out').querySelectorAll('.more').forEach(btn => {
        btn.addEventListener('click', () => {
          const box = $('w' + btn.dataset.i);
          box.hidden = !box.hidden;
          btn.textContent = box.hidden ? 'show surrounding' : 'hide surrounding';
        });
      });
    }
  } catch (err) {
    $('out').innerHTML = `<p class="status">Error: ${esc(String(err))}</p>`;
  } finally {
    $('go').disabled = false;
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
