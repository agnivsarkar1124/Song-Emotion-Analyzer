"""
Self-contained HTML/CSS/JS view for a mood playlist (the dict from MoodIndex.search()).

``render_playlist_html(playlist)`` returns one HTML document meant for
``streamlit.components.v1.html``. It has four tabs:

    Playlist   scrollable, colour-coded cards; click one to expand scores, an
               emotion mix, a 30-second preview player and search links
    Mood Map   valence x intensity scatter with the query as a star; click a dot
               to jump to its card
    Breakdown  how the description was interpreted, and query vs. playlist emotions
    JSON       the raw playlist with copy / download buttons

All dynamic text is inserted with textContent / createTextNode (never innerHTML),
and links only open for ``https://`` URLs, because titles and URLs originate from
scraped or third-party data.
"""

from __future__ import annotations

import json

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mood playlist</title>
<style>
  :root {
    --bg1:#12122b; --bg2:#1d1240; --bg3:#2b1050;
    --text:#f4f4ff; --muted:#a9a9cf; --line:rgba(255,255,255,.12);
    --card:rgba(255,255,255,.07); --card-hover:rgba(255,255,255,.12);
  }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; }
  body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         color:var(--text); background:transparent; }
  .app { height:100%; display:flex; flex-direction:column; border-radius:18px; overflow:hidden;
         background: radial-gradient(1200px 500px at 0% 0%, #3b1c7a55, transparent 60%),
                     radial-gradient(900px 500px at 100% 100%, #0e8f8a33, transparent 60%),
                     linear-gradient(160deg, var(--bg1), var(--bg2) 55%, var(--bg3)); }
  header { padding:18px 22px 0; }
  h1 { margin:0; font-size:20px; letter-spacing:.2px; }
  .query { margin:4px 0 0; color:var(--muted); font-size:14px; font-style:italic; overflow-wrap:anywhere; }
  .meta { margin:6px 0 0; font-size:12px; color:var(--muted); }
  [role=tablist] { display:flex; gap:6px; padding:14px 18px 0; border-bottom:1px solid var(--line); overflow-x:auto; }
  [role=tab] { appearance:none; border:0; background:transparent; color:var(--muted); font:inherit; font-size:14px;
               padding:10px 14px; border-radius:10px 10px 0 0; cursor:pointer; white-space:nowrap; position:relative; }
  [role=tab]:hover { color:var(--text); background:rgba(255,255,255,.06); }
  [role=tab][aria-selected=true] { color:var(--text); background:rgba(255,255,255,.1); font-weight:600; }
  [role=tab][aria-selected=true]::after { content:""; position:absolute; left:10px; right:10px; bottom:-1px; height:3px;
               border-radius:3px; background:linear-gradient(90deg,#ff9f1c,#f15bb5,#4f7cff); }
  [role=tab]:focus-visible, .card:focus-visible, button:focus-visible, a:focus-visible { outline:2px solid #fff; outline-offset:2px; }
  .panel { flex:1; overflow-y:auto; padding:16px 18px 22px; scroll-behavior:smooth; }
  .panel[hidden] { display:none; }
  .panel::-webkit-scrollbar { width:10px; } .panel::-webkit-scrollbar-thumb { background:rgba(255,255,255,.22); border-radius:10px; }

  /* cards */
  .card { position:relative; display:grid; grid-template-columns:auto 1fr auto; gap:6px 14px; align-items:center;
          padding:14px 16px 14px 20px; margin-bottom:10px; border-radius:16px; background:var(--card);
          border:1px solid var(--line); cursor:pointer; overflow:hidden; transition:background .15s, transform .15s, box-shadow .15s; }
  .card:hover { background:var(--card-hover); transform:translateY(-1px); }
  .card .bar { position:absolute; left:0; top:0; bottom:0; width:6px; }
  .card.open { background:var(--card-hover); box-shadow:0 8px 28px rgba(0,0,0,.35); }
  .card.flash { animation:flash 1.2s ease-out; }
  @keyframes flash { 0% { box-shadow:0 0 0 3px #fff; } 100% { box-shadow:0 0 0 0 transparent; } }
  .rank { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; font-weight:700; font-size:17px; color:#fff; }
  .title { font-weight:650; font-size:16px; overflow-wrap:anywhere; }
  .artist { color:var(--muted); font-size:13px; margin-top:1px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:7px; }
  .chip { font-size:11.5px; padding:3px 9px; border-radius:999px; background:rgba(255,255,255,.1); border:1px solid var(--line); }
  .chip.emo { color:#fff; border:0; font-weight:600; }
  .ring { --p:0; width:54px; height:54px; border-radius:50%; display:grid; place-items:center; font-weight:700; font-size:13px;
          background:conic-gradient(var(--c1) calc(var(--p) * 1%), rgba(255,255,255,.14) 0); position:relative; }
  .ring::before { content:""; position:absolute; inset:5px; border-radius:50%; background:#1a1238; }
  .ring span { position:relative; }
  .detail { grid-column:1 / -1; display:none; margin-top:10px; padding-top:12px; border-top:1px dashed var(--line); cursor:default; }
  .card.open .detail { display:block; }
  .why { margin:0 0 12px; font-size:14px; line-height:1.45; }
  .scores { display:grid; gap:8px; margin-bottom:12px; }
  .score { display:grid; grid-template-columns:150px 1fr 42px; gap:10px; align-items:center; font-size:12.5px; color:var(--muted); }
  .track { height:8px; border-radius:8px; background:rgba(255,255,255,.12); overflow:hidden; }
  .fill { height:100%; border-radius:8px; }
  .stack { display:flex; height:12px; border-radius:8px; overflow:hidden; background:rgba(255,255,255,.1); }
  .legend { display:flex; flex-wrap:wrap; gap:6px 12px; margin:8px 0 12px; font-size:12px; color:var(--muted); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
  .actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .btn { display:inline-block; font-size:13px; padding:7px 13px; border-radius:10px; text-decoration:none; color:#fff; border:0; cursor:pointer;
         background:rgba(255,255,255,.14); }
  .btn:hover { background:rgba(255,255,255,.24); }
  audio { height:34px; max-width:100%; }
  .faint { color:var(--muted); font-size:12.5px; }

  /* map */
  .mapwrap { background:rgba(0,0,0,.2); border:1px solid var(--line); border-radius:16px; padding:8px; }
  svg { width:100%; height:auto; display:block; }
  .pt { cursor:pointer; } .pt:hover circle { stroke:#fff; stroke-width:3; }
  .help { color:var(--muted); font-size:12.5px; margin:10px 2px 0; }

  /* breakdown */
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); margin:18px 0 8px; }
  h2:first-child { margin-top:2px; }
  .chip.concept { background:linear-gradient(135deg,#4f7cff55,#f15bb555); }
  .chip.neg { text-decoration:line-through; opacity:.8; }
  .meter { position:relative; height:10px; margin:22px 4px 26px; border-radius:10px;
           background:linear-gradient(90deg,#3d4a8f,#55557a 50%,#ffb800); }
  .meter.int { background:linear-gradient(90deg,#2ec4b6,#55557a 50%,#ff4d4d); }
  .mark { position:absolute; top:-6px; width:22px; height:22px; margin-left:-11px; border-radius:50%; border:3px solid #fff; display:grid;
          place-items:center; font-size:10px; font-weight:700; }
  .mark small { position:absolute; top:24px; font-size:10.5px; font-weight:500; color:var(--muted); white-space:nowrap; }
  .ends { display:flex; justify-content:space-between; font-size:11.5px; color:var(--muted); margin:-16px 4px 18px; }
  .erow { display:grid; grid-template-columns:100px 1fr; gap:10px; align-items:center; margin:7px 0; font-size:13px; }
  .bars { display:grid; gap:3px; }
  .b { height:8px; border-radius:8px; min-width:2px; }

  /* json */
  pre { margin:0; padding:14px; border-radius:14px; background:rgba(0,0,0,.35); border:1px solid var(--line);
        font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; overflow:auto; white-space:pre; color:#e9e9ff; }
  .toolbar { display:flex; gap:8px; margin-bottom:10px; align-items:center; }
  @media (max-width:560px) { .score { grid-template-columns:100px 1fr 38px; } .ring { width:46px; height:46px; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>🎧 Your mood playlist</h1>
    <p class="query" id="query"></p>
    <p class="meta" id="meta"></p>
  </header>
  <div role="tablist" aria-label="Playlist views" id="tabs"></div>
  <section class="panel" id="p-playlist" role="tabpanel"></section>
  <section class="panel" id="p-map" role="tabpanel" hidden></section>
  <section class="panel" id="p-breakdown" role="tabpanel" hidden></section>
  <section class="panel" id="p-json" role="tabpanel" hidden></section>
</div>

<script id="playlist-data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("playlist-data").textContent);
  const EMOTIONS = ["joy","trust","anticipation","surprise","fear","sadness","disgust","anger"];
  const PALETTE = {
    joy:["#FFB800","#FF7A00"], trust:["#2EC4B6","#0E8F8A"], anticipation:["#FF9F1C","#F15B2A"],
    surprise:["#F15BB5","#9B5DE5"], fear:["#9B5DE5","#5B2A9D"], sadness:["#4F7CFF","#2B3FD6"],
    disgust:["#9BD43A","#4C9A2A"], anger:["#FF4D4D","#C2185B"], none:["#8892b0","#5b6486"]
  };
  const pal = e => PALETTE[e] || PALETTE.none;
  const grad = e => `linear-gradient(135deg, ${pal(e)[0]}, ${pal(e)[1]})`;
  const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
  const pct = x => Math.round(100 * x) + "%";
  const isHttps = u => typeof u === "string" && u.startsWith("https://");
  const fmtDur = s => (s == null ? null : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"));

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  const tracks = DATA.tracks || [];
  const pm = DATA.parsed_mood || {};

  // ---- header ---------------------------------------------------------------------
  document.getElementById("query").textContent = "\u201C" + DATA.query + "\u201D";
  const avg = tracks.length ? Math.round(tracks.reduce((a, t) => a + t.match_percent, 0) / tracks.length) : 0;
  document.getElementById("meta").textContent =
    `Top ${tracks.length} of ${DATA.database_size} songs \u00B7 average match ${avg}% \u00B7 ranked by semantic distance`;

  // ---- tabs -----------------------------------------------------------------------
  const TABS = [["playlist", "🎵 Playlist"], ["map", "🗺️ Mood map"], ["breakdown", "📊 Breakdown"], ["json", "{ } JSON"]];
  const tabsEl = document.getElementById("tabs");
  function showTab(name) {
    for (const [id] of TABS) {
      const on = id === name;
      document.getElementById("t-" + id).setAttribute("aria-selected", String(on));
      document.getElementById("t-" + id).tabIndex = on ? 0 : -1;
      document.getElementById("p-" + id).hidden = !on;
    }
  }
  TABS.forEach(([id, label], i) => {
    tabsEl.append(h("button", {
      role: "tab", id: "t-" + id, "aria-controls": "p-" + id, "aria-selected": String(i === 0), tabindex: i === 0 ? 0 : -1,
      onclick: () => showTab(id),
      onkeydown: ev => {
        const dir = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
        if (!dir) return;
        const next = TABS[(i + dir + TABS.length) % TABS.length][0];
        showTab(next); document.getElementById("t-" + next).focus();
      }
    }, label));
  });

  // ---- playlist tab -----------------------------------------------------------------
  const cards = [];
  const panelPlaylist = document.getElementById("p-playlist");

  function scoreRow(label, value, color) {
    return h("div", { class: "score" }, h("span", {}, label),
      h("div", { class: "track" }, h("div", { class: "fill", style: `width:${Math.round(value * 100)}%;background:${color}` })),
      h("span", {}, pct(value)));
  }

  function emotionStack(profile) {
    const stack = h("div", { class: "stack", role: "img", "aria-label": "Emotion mix" });
    const entries = EMOTIONS.map(e => [e, profile[e] || 0]).filter(([, v]) => v > 0.001);
    for (const [e, v] of entries) stack.append(h("div", { title: `${cap(e)} ${pct(v)}`, style: `width:${v * 100}%;background:${pal(e)[0]}` }));
    const top = entries.slice().sort((a, b) => b[1] - a[1]).slice(0, 4);
    const legend = h("div", { class: "legend" }, top.map(([e, v]) =>
      h("span", {}, h("i", { class: "dot", style: `background:${pal(e)[0]}` }), `${cap(e)} ${pct(v)}`)));
    return [stack, legend];
  }

  function buildDetail(t) {
    const d = h("div", { class: "detail" });
    d.addEventListener("click", ev => ev.stopPropagation());
    d.append(h("p", { class: "why" }, t.why));
    d.append(h("div", { class: "scores" },
      scoreRow("Emotion fit", t.scores.emotion, "linear-gradient(90deg,#f15bb5,#9b5de5)"),
      scoreRow("Mood fit (tone + swing)", t.scores.mood, "linear-gradient(90deg,#4f7cff,#2ec4b6)"),
      scoreRow("Lyric themes", t.scores.theme, "linear-gradient(90deg,#ffb800,#ff7a00)")));
    d.append(h("div", { class: "faint" }, "Emotion mix in the lyrics"));
    d.append(...emotionStack(t.emotion_profile || {}));
    const facts = [
      `Distance ${t.distance.toFixed(3)}`, `Tone ${t.valence >= 0 ? "+" : ""}${t.valence.toFixed(2)}`,
      `Swing ${t.intensity >= 0 ? "+" : ""}${t.intensity.toFixed(2)}`
    ];
    if (t.emotion_confidence != null && t.top_emotion) facts.push(`Emotion confidence ${pct(t.emotion_confidence)}`);
    d.append(h("p", { class: "faint" }, facts.join("  \u00B7  ")));
    const actions = h("div", { class: "actions" });
    const q = encodeURIComponent(t.title + " " + t.artist);
    if (isHttps(t.preview_url)) actions.append(h("audio", { controls: "", preload: "none", src: t.preview_url }));
    else actions.append(h("span", { class: "faint" }, "No preview clip available."));
    if (isHttps(t.track_url)) actions.append(h("a", { class: "btn", href: t.track_url, target: "_blank", rel: "noopener noreferrer" }, "Open in Apple Music"));
    actions.append(h("a", { class: "btn", href: "https://open.spotify.com/search/" + q, target: "_blank", rel: "noopener noreferrer" }, "Search Spotify"));
    actions.append(h("a", { class: "btn", href: "https://www.youtube.com/results?search_query=" + q, target: "_blank", rel: "noopener noreferrer" }, "Search YouTube"));
    d.append(actions);
    return d;
  }

  function setOpen(card, open) {
    card.classList.toggle("open", open);
    card.setAttribute("aria-expanded", String(open));
    if (!open) card.querySelectorAll("audio").forEach(a => a.pause());
  }

  function toggleCard(idx, opts) {
    const card = cards[idx];
    const willOpen = opts && opts.forceOpen ? true : !card.classList.contains("open");
    cards.forEach((c, j) => { if (j !== idx) setOpen(c, false); });
    setOpen(card, willOpen);
    if (willOpen && opts && opts.scroll) card.scrollIntoView({ block: "nearest" });
    if (opts && opts.flash) { card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
  }

  tracks.forEach((t, idx) => {
    const emo = t.top_emotion;
    const chips = h("div", { class: "chips" });
    if (emo) chips.append(h("span", { class: "chip emo", style: `background:${grad(emo)}` }, cap(emo)));
    if (t.secondary_emotion && emo) chips.append(h("span", { class: "chip" }, "then " + t.secondary_emotion));
    if (t.category) chips.append(h("span", { class: "chip" }, t.category));
    const dur = fmtDur(t.duration_seconds);
    if (dur) chips.append(h("span", { class: "chip" }, dur));

    const card = h("div", {
      class: "card", role: "button", tabindex: 0, "aria-expanded": "false",
      onclick: () => toggleCard(idx),
      onkeydown: ev => { if ((ev.key === "Enter" || ev.key === " ") && ev.target === card) { ev.preventDefault(); toggleCard(idx); } }
    },
      h("div", { class: "bar", style: `background:${grad(emo)}` }),
      h("div", { class: "rank", style: `background:${grad(emo)}` }, t.rank),
      h("div", {}, h("div", { class: "title" }, t.title), h("div", { class: "artist" }, t.artist), chips),
      h("div", { class: "ring", style: `--p:${t.match_percent};--c1:${pal(emo)[0]}`, title: `${t.match_percent}% match` }, h("span", {}, t.match_percent + "%")),
      buildDetail(t));
    cards.push(card);
    panelPlaylist.append(card);
  });
  if (!tracks.length) panelPlaylist.append(h("p", { class: "help" }, "No tracks to show."));

  // ---- mood map tab -----------------------------------------------------------------
  (function buildMap() {
    const panel = document.getElementById("p-map");
    const W = 640, H = 400, M = 44, LIM = 1.6;
    const X = v => M + (v + LIM) / (2 * LIM) * (W - 2 * M);
    const Y = v => H - M - (v + LIM) / (2 * LIM) * (H - 2 * M);
    const NS = "http://www.w3.org/2000/svg";
    const s = (tag, attrs, text) => {
      const el = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
      if (text !== undefined) el.textContent = text;
      return el;
    };
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Mood map of the playlist" });
    const quad = (x, y, w, hh, fill) => svg.append(s("rect", { x, y, width: w, height: hh, fill, rx: 10 }));
    quad(M, M, X(0) - M, Y(0) - M, "rgba(155,93,229,.12)");
    quad(X(0), M, W - M - X(0), Y(0) - M, "rgba(255,122,0,.12)");
    quad(M, Y(0), X(0) - M, H - M - Y(0), "rgba(79,124,255,.14)");
    quad(X(0), Y(0), W - M - X(0), H - M - Y(0), "rgba(46,196,182,.13)");
    const lab = (x, y, t, anchor) => svg.append(s("text", { x, y, fill: "#bdbde6", "font-size": 12, "text-anchor": anchor || "start", opacity: .85 }, t));
    lab(M + 8, M + 18, "volatile & dark"); lab(W - M - 8, M + 18, "volatile & bright", "end");
    lab(M + 8, Y(0) + 18, "calm & dark"); lab(W - M - 8, Y(0) + 18, "calm & bright", "end");
    svg.append(s("line", { x1: X(0), y1: M, x2: X(0), y2: H - M, stroke: "rgba(255,255,255,.3)", "stroke-dasharray": "4 4" }));
    svg.append(s("line", { x1: M, y1: Y(0), x2: W - M, y2: Y(0), stroke: "rgba(255,255,255,.3)", "stroke-dasharray": "4 4" }));
    svg.append(s("text", { x: W / 2, y: H - 10, fill: "#a9a9cf", "font-size": 12, "text-anchor": "middle" }, "valence: darker \u2192 brighter"));
    svg.append(s("text", { x: 14, y: H / 2, fill: "#a9a9cf", "font-size": 12, "text-anchor": "middle", transform: `rotate(-90 14 ${H / 2})` }, "intensity: steadier \u2192 more volatile"));

    const qx = X(Math.max(-LIM, Math.min(LIM, pm.valence || 0))), qy = Y(Math.max(-LIM, Math.min(LIM, pm.intensity || 0)));
    tracks.forEach(t => {
      const x = X(Math.max(-LIM, Math.min(LIM, t.valence))), y = Y(Math.max(-LIM, Math.min(LIM, t.intensity)));
      svg.append(s("line", { x1: qx, y1: qy, x2: x, y2: y, stroke: "rgba(255,255,255,.28)", "stroke-dasharray": "3 4" }));
    });
    tracks.forEach((t, idx) => {
      const x = X(Math.max(-LIM, Math.min(LIM, t.valence))), y = Y(Math.max(-LIM, Math.min(LIM, t.intensity)));
      const g = s("g", { class: "pt", tabindex: 0, role: "button", "aria-label": `${t.rank}. ${t.title} by ${t.artist}` });
      g.append(s("circle", { cx: x, cy: y, r: 14, fill: pal(t.top_emotion)[0], stroke: "#fff", "stroke-width": 1.5 }));
      g.append(s("text", { x, y: y + 4.5, fill: "#fff", "font-size": 13, "font-weight": 700, "text-anchor": "middle", "pointer-events": "none" }, String(t.rank)));
      g.append(s("title", {}, `${t.rank}. ${t.title} \u2013 ${t.artist}  (${t.match_percent}% match)`));
      const go = () => { showTab("playlist"); toggleCard(idx, { forceOpen: true, scroll: true, flash: true }); };
      g.addEventListener("click", go);
      g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); } });
      svg.append(g);
    });
    const star = s("path", { d: "M0,-15 L4.4,-5 L15,-4.6 L6.8,2.3 L9.3,12.7 L0,7 L-9.3,12.7 L-6.8,2.3 L-15,-4.6 L-4.4,-5 Z",
      transform: `translate(${qx} ${qy})`, fill: "#fff", stroke: "#ffb800", "stroke-width": 2 });
    star.append(s("title", {}, "Your description"));
    svg.append(star);
    svg.append(s("text", { x: qx, y: qy - 20, fill: "#fff", "font-size": 12, "font-weight": 700, "text-anchor": "middle" }, "your mood"));

    panel.append(h("div", { class: "mapwrap" }, svg));
    panel.append(h("p", { class: "help" }, "Each dot is a track placed by its lyrics\u2019 tone and emotional swing; the star is your description. Shorter dashed lines mean a closer match. Click a dot to open its card."));
  })();

  // ---- breakdown tab ------------------------------------------------------------------
  (function buildBreakdown() {
    const panel = document.getElementById("p-breakdown");
    panel.append(h("h2", {}, "How your description was read"));
    const concepts = h("div", { class: "chips" });
    (pm.concepts || []).forEach(c => concepts.append(h("span", { class: "chip concept" + (c.negated ? " neg" : "") },
      c.term + (c.weight !== 1 ? " \u00D7" + c.weight : ""))));
    if (!(pm.concepts || []).length) concepts.append(h("span", { class: "faint" }, "No known mood words \u2013 sentiment lexicon only."));
    panel.append(concepts);
    if ((pm.notes || []).length) pm.notes.forEach(n => panel.append(h("p", { class: "help" }, n)));
    if ((pm.keywords || []).length) {
      panel.append(h("h2", {}, "Lyric keywords searched"));
      panel.append(h("div", { class: "chips" }, pm.keywords.map(k => h("span", { class: "chip" }, k))));
    }

    const meter = (title, cls, qv, pv, lo, hi) => {
      const pos = v => (Math.max(-1.5, Math.min(1.5, v)) + 1.5) / 3 * 100 + "%";
      const m = h("div", { class: "meter " + cls },
        h("div", { class: "mark", style: `left:${pos(qv)};background:#ffb800` }, "\u2605", h("small", {}, "you")),
        h("div", { class: "mark", style: `left:${pos(pv)};background:#2b3fd6;top:-6px` }, "", h("small", { style: "top:-18px" }, "playlist avg")));
      panel.append(h("h2", {}, title), m, h("div", { class: "ends" }, h("span", {}, lo), h("span", {}, hi)));
    };
    meter("Tone", "val", pm.valence || 0, (DATA.summary || {}).mean_valence || 0, "darker", "brighter");
    meter("Emotional swing", "int", pm.intensity || 0, (DATA.summary || {}).mean_intensity || 0, "steadier", "more volatile");

    panel.append(h("h2", {}, "Emotions: your description vs. the playlist"));
    const q = pm.emotions || {}, p = (DATA.summary || {}).emotion_profile || {};
    const maxv = Math.max(0.01, ...EMOTIONS.map(e => Math.max(q[e] || 0, p[e] || 0)));
    EMOTIONS.forEach(e => panel.append(h("div", { class: "erow" },
      h("span", {}, h("i", { class: "dot", style: `background:${pal(e)[0]}` }), cap(e)),
      h("div", { class: "bars" },
        h("div", { class: "b", title: "You: " + pct(q[e] || 0), style: `width:${(q[e] || 0) / maxv * 100}%;background:rgba(255,255,255,.75)` }),
        h("div", { class: "b", title: "Playlist: " + pct(p[e] || 0), style: `width:${(p[e] || 0) / maxv * 100}%;background:${grad(e)}` })))));
    panel.append(h("p", { class: "help" }, "White bar = emotions implied by your words. Coloured bar = average emotion mix of the playlist\u2019s lyrics."));

    panel.append(h("h2", {}, "How distance is computed"));
    const w = DATA.weights || {};
    panel.append(h("div", { class: "chips" },
      h("span", { class: "chip" }, "emotion " + pct(w.emotion || 0)),
      h("span", { class: "chip" }, "tone + swing " + pct(w.mood || 0)),
      h("span", { class: "chip" }, "lyric themes " + pct(w.theme || 0))));
    panel.append(h("p", { class: "help" }, "distance = 1 \u2212 (weighted blend of emotion-profile cosine, a Gaussian kernel on tone/swing, and TF-IDF theme similarity). Lower is closer."));
  })();

  // ---- JSON tab -------------------------------------------------------------------------
  (function buildJson() {
    const panel = document.getElementById("p-json");
    const text = JSON.stringify(DATA, null, 2);
    const status = h("span", { class: "faint" }, "");
    const flash = msg => { status.textContent = msg; setTimeout(() => { status.textContent = ""; }, 2000); };
    const copy = () => {
      const fallback = () => {
        const ta = h("textarea", { style: "position:fixed;opacity:0" }); ta.value = text; document.body.append(ta); ta.select();
        try { document.execCommand("copy"); flash("Copied"); } catch (e) { flash("Copy blocked \u2013 select the text instead"); }
        ta.remove();
      };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => flash("Copied"), fallback);
      else fallback();
    };
    const download = () => {
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const a = h("a", { href: url, download: "mood_playlist.json" });
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    panel.append(h("div", { class: "toolbar" },
      h("button", { class: "btn", onclick: copy }, "Copy JSON"),
      h("button", { class: "btn", onclick: download }, "Download .json"), status));
    panel.append(h("pre", {}, text));
  })();

  if (tracks.length) toggleCard(0, {});   // start with the best match expanded
})();
</script>
</body>
</html>
"""


def render_playlist_html(playlist: dict) -> str:
    """Return a complete HTML document rendering ``playlist``.

    The JSON is embedded in a ``<script type="application/json">`` block with every
    ``<`` escaped as ``\\u003c`` so no string in it (e.g. a scraped song title) can
    close the script tag or open an HTML comment.
    """
    payload = json.dumps(playlist, ensure_ascii=False).replace("<", "\\u003c")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return _TEMPLATE.replace("__DATA__", payload)
