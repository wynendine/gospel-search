"use client";

import { useState } from "react";

type Hit = {
  citation: string;
  url: string;
  speaker: string;
  year: string;
  kind: string;
  title: string;
  text: string;
  window: string;
  rerank: number | null;
  dense: number | null;
  lexical: number | null;
};

export default function Home() {
  const [q, setQ] = useState("");
  const [source, setSource] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [after, setAfter] = useState("");
  const [before, setBefore] = useState("");
  const [wantAnswer, setWantAnswer] = useState(true);
  const [rerank, setRerank] = useState(true);
  const [hyde, setHyde] = useState(true);

  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Set<number>>(new Set());

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;

    setBusy(true);
    setError("");
    setAnswer("");
    setHits(null);
    setOpen(new Set());

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q.trim(),
          n: 10,
          speaker: speaker.trim() || null,
          after: after ? Number(after) : null,
          before: before ? Number(before) : null,
          source: source || null,
          answer: wantAnswer,
          rerank,
          hyde,
        }),
      });
      if (!res.ok) throw new Error(`Search failed (${res.status})`);
      const data = await res.json();
      setAnswer(data.answer ?? "");
      setHits(data.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function toggle(i: number) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }

  return (
    <>
      <header>
        <h1>Gospel Search</h1>
        <p className="sub">
          General Conference talks and the standard works — searched by meaning,
          not keyword.
        </p>
      </header>

      <main>
        <form className="search" onSubmit={submit}>
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Describe what was said…"
            autoFocus
            autoComplete="off"
          />
          <button type="submit" disabled={busy || !q.trim()}>
            {busy ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="filters">
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">everything</option>
            <option value="talks">talks only</option>
            <option value="scriptures">scriptures only</option>
          </select>
          <input
            type="text"
            value={speaker}
            onChange={(e) => setSpeaker(e.target.value)}
            placeholder="speaker"
          />
          <input
            type="number"
            value={after}
            onChange={(e) => setAfter(e.target.value)}
            placeholder="after"
          />
          <input
            type="number"
            value={before}
            onChange={(e) => setBefore(e.target.value)}
            placeholder="before"
          />
          <label className="check">
            <input
              type="checkbox"
              checked={wantAnswer}
              onChange={(e) => setWantAnswer(e.target.checked)}
            />
            answer
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
            />
            rerank
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={hyde}
              onChange={(e) => setHyde(e.target.checked)}
            />
            hyde
          </label>
        </div>

        {answer && <div className="answer">{answer}</div>}
        {error && <p className="status">Error: {error}</p>}
        {busy && <p className="status">Searching…</p>}
        {hits?.length === 0 && <p className="status">No matches.</p>}

        {hits?.map((h, i) => {
          const bits = [
            h.speaker,
            h.year,
            h.rerank !== null ? `rerank ${Math.round(h.rerank)}/10` : null,
            [h.dense && "dense", h.lexical && "bm25"].filter(Boolean).join("+") ||
              null,
          ].filter(Boolean);

          return (
            <div className="card" key={`${h.url}-${i}`}>
              <a
                className="cite"
                href={h.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                {i + 1}. {h.citation}
              </a>
              <div className="meta">{bits.join(" · ")}</div>
              <p className="passage">{h.text}</p>
              <button className="more" onClick={() => toggle(i)}>
                {open.has(i) ? "hide surrounding" : "show surrounding"}
              </button>
              {open.has(i) && <div className="window">{h.window}</div>}
            </div>
          );
        })}
      </main>
    </>
  );
}
