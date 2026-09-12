"""Local web UI — one page, served from localhost."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import answer as answer_mod
from . import cache as cache_mod
from . import index as idx
from . import research as research_mod
from . import search as search_mod
from .plan import Plan, plan as make_plan

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
    plan: bool = True
    fresh: bool = False


@app.get("/api/history")
def api_history(limit: int = 40):
    return {"queries": cache_mod.recent(limit)}


@app.post("/api/search")
def api_search(request: Query):
    conn, vectors = _resources()
    filters = search_mod.Filters(
        speaker=request.speaker,
        after=request.after,
        before=request.before,
        source=request.source,
    )

    # Same shape as the CLI: a bare reference resolves directly, otherwise the
    # planner picks the retrieval strategy. The UI used to call search()
    # straight through, which meant none of the intents reached the browser.
    findings = research_mod.resolve_reference(conn, request.query)

    # References are already free; everything else is worth caching, which also
    # gives the history list something to replay.
    ckey = None
    if findings is None:
        flags = {
            "n": request.n, "hyde": request.hyde, "rerank": request.rerank,
            "answer": request.answer, "plan": request.plan,
        }
        ckey = cache_mod.key(request.query, filters, flags)
        if not request.fresh:
            hit = cache_mod.get(ckey)
            if hit:
                hit["cached"] = True
                return hit

    if findings is None:
        plan = (
            Plan(topic=request.query, filters=filters)
            if not request.plan
            else make_plan(request.query, override=filters)
        )
        findings = research_mod.investigate(
            request.query,
            plan,
            n=request.n,
            use_hyde=request.hyde,
            use_rerank=request.rerank,
            conn=conn,
            vectors=vectors,
        )

    results = findings.results
    text = ""
    if request.answer and results:
        text = answer_mod.answer(request.query, findings, conn=conn)

    payload = {
        "answer": text,
        "intent": findings.plan.intent,
        "coverage": findings.coverage.summary(),
        "exhaustive": findings.exhaustive,
        "total_matches": findings.total_matches,
        "breakdown": findings.breakdown,
        "breakdown_label": findings.breakdown_label,
        "note": findings.note,
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
    if ckey:
        cache_mod.put(ckey, request.query, payload)
    return payload


@app.get("/api/stats")
def api_stats():
    return idx.read_meta()


PAGE = r"""<!doctype html>
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
    border-radius: 10px; padding: 4px 20px 14px; margin-bottom: 26px;
    font-size: 15.5px;
  }
  #answerBox h2 {
    font-size: 15px; margin: 18px 0 6px; letter-spacing: -0.01em;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  #answerBox h2:first-child { margin-top: 14px; }
  #answerBox p { margin: 0 0 10px; }
  #answerBox ul { margin: 0 0 10px; padding-left: 20px; }
  #answerBox li { margin: 3px 0; }
  #answerBox code {
    font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px;
    background: var(--panel); padding: 1px 4px; border-radius: 3px;
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
  .hist { font-family: ui-sans-serif, system-ui, sans-serif; margin-bottom: 22px; }
  .hist > summary {
    cursor: pointer; font-size: 12px; color: var(--muted);
    list-style: none; user-select: none; padding: 2px 0;
  }
  .hist > summary::-webkit-details-marker { display: none; }
  .hist > summary::before { content: "\25B8 "; }
  .hist[open] > summary::before { content: "\25BE "; }
  .hist ol { list-style: none; margin: 8px 0 0; padding: 0; }
  .hist li { display: flex; align-items: baseline; gap: 8px; padding: 3px 0; }
  .hist button.q {
    background: none; border: 0; padding: 0; cursor: pointer; text-align: left;
    font: inherit; font-size: 13px; font-weight: 400; color: var(--accent);
    text-decoration: none; flex: 1; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .hist button.q:hover { text-decoration: underline; }
  .hist .k {
    font-size: 10.5px; color: var(--muted); border: 1px solid var(--line);
    border-radius: 999px; padding: 0 6px; flex: none;
  }
  .hist .ago { font-size: 11px; color: var(--muted); flex: none; }
  .hist .stale { opacity: .55; }
  .coverage {
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    color: var(--muted); margin-bottom: 18px;
  }
  .chip {
    display: inline-block; background: var(--accent-soft); color: var(--accent);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 1px 9px; margin-right: 7px; font-weight: 600;
  }
  .note {
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12.5px;
    color: var(--muted); border-left: 2px solid var(--accent);
    padding: 2px 0 2px 10px; margin: 0 0 18px;
  }
  .dist { margin: 0 0 22px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; }
  .dist div { display: flex; align-items: center; gap: 8px; margin: 2px 0; }
  .dist span.k { width: 150px; color: var(--muted); text-align: right; }
  .dist span.n { width: 56px; color: var(--muted); }
  .dist i { background: var(--accent); height: 9px; border-radius: 2px; display: block; }
</style>
</head>
<body>
<header>
  <h1>Gospel Search</h1>
  <p class="sub">General Conference talks and the standard works &mdash; searched by meaning, not keyword.</p>
</header>
<main>
  <form id="f">
    <input type="text" id="q" placeholder="Ask a question, describe a passage, or type a reference&hellip;" autofocus autocomplete="off">
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

  <details class="hist" id="hist" hidden>
    <summary id="histLabel">Recent searches</summary>
    <ol id="histList"></ol>
  </details>

  <div id="answerBox" hidden></div>
  <div id="coverage" class="coverage" hidden></div>
  <div id="note" class="note" hidden></div>
  <div id="dist" class="dist" hidden></div>
  <div id="out"></div>
</main>

<script>
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = el => el.value ? parseInt(el.value, 10) : null;

// The per-intent prompts produce structured answers — headings for a thematic
// survey, lists for an enumeration — so the panel has to render Markdown
// rather than show the asterisks. Escape first, format second: nothing the
// model writes can become live markup.
function md(src) {
  const inline = t => esc(t)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  const out = [];
  let list = null;
  for (const raw of src.split('\n')) {
    const line = raw.trimEnd();
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);

    if (bullet) { (list ??= []).push(inline(bullet[1])); continue; }
    if (list) { out.push('<ul>' + list.map(i => `<li>${i}</li>`).join('') + '</ul>'); list = null; }
    if (heading) { out.push(`<h2>${inline(heading[1])}</h2>`); continue; }
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push('<ul>' + list.map(i => `<li>${i}</li>`).join('') + '</ul>');
  return out.join('');
}

const ago = t => {
  const s = Date.now() / 1000 - t;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
};

// History is the query cache read back, so replaying an entry is free and
// instant rather than a second paid search.
async function loadHistory() {
  try {
    const res = await fetch('/api/history?limit=40');
    if (!res.ok) return;
    const { queries } = await res.json();
    if (!queries.length) { $('hist').hidden = true; return; }

    $('histLabel').textContent = `Recent searches (${queries.length})`;
    $('histList').innerHTML = queries.map((h, i) => `
      <li class="${h.fresh ? '' : 'stale'}">
        <span class="k">${esc(h.intent || '?')}</span>
        <button class="q" data-i="${i}" title="${esc(h.query)}">${esc(h.query)}</button>
        <span class="ago">${ago(h.at)}</span>
      </li>`).join('');

    $('histList').querySelectorAll('button.q').forEach(b => {
      b.addEventListener('click', () => {
        $('q').value = queries[b.dataset.i].query;
        $('f').requestSubmit();
      });
    });
    $('hist').hidden = false;
  } catch { /* history is a convenience; never let it break the page */ }
}

document.addEventListener('DOMContentLoaded', loadHistory);

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
      $('answerBox').innerHTML = md(data.answer);
      $('answerBox').hidden = false;
    }

    // Which strategy ran, and what the answer actually got to see.
    let cov = `<span class="chip">${esc(data.intent || 'lookup')}</span>`;
    if (data.exhaustive && data.total_matches != null) {
      cov += `complete: all ${data.total_matches.toLocaleString()} matches &middot; `;
    } else if (data.total_matches != null) {
      cov += `${data.total_matches.toLocaleString()} matches, showing examples &middot; `;
    }
    cov += esc(data.coverage || '');
    $('coverage').innerHTML = cov;
    $('coverage').hidden = false;

    if (data.note) {
      $('note').textContent = data.note;
      $('note').hidden = false;
    } else { $('note').hidden = true; }

    if (data.breakdown && data.breakdown.length) {
      const max = Math.max(...data.breakdown.map(r => r[1]));
      $('dist').innerHTML =
        `<p style="margin:0 0 6px"><strong>${esc(data.breakdown_label || '')}</strong></p>` +
        data.breakdown.map(([k, n]) =>
          `<div><span class="k">${esc(String(k))}</span>` +
          `<i style="width:${Math.max(2, 240 * n / max)}px"></i>` +
          `<span class="n">${n.toLocaleString()}</span></div>`).join('');
      $('dist').hidden = false;
    } else { $('dist').hidden = true; }

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
    loadHistory();
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
