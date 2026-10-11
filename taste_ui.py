"""
Interactive "mood mix" view: a two-way Streamlit component (``st.components.v2``).

Tabs: For you (personalised cards with like / nope / save / skip), Rate (a swipe
deck of the songs the model is least sure about), Your taste (insights, the
emotion x energy grid, theme leanings, how the profile evolved), Compare
(original vs personalised ranking as a slopegraph), Saved, and JSON.

Clicks are sent back to Python as a ``feedback`` trigger:
``{"action": "like"|"dislike"|"save"|"unsave"|"skip"|"clear", "key": song key, "nonce": ...}``.

Security: v2 components run in the app page (style-isolated in a shadow root, but
not sandboxed), and song titles come from scraped pages. So the JS builds every
node with ``textContent`` / ``createTextNode`` - never ``innerHTML`` with data -
and only follows ``https://`` links.
"""

from __future__ import annotations

from emotion_engine import EMOTIONS
from preferences import EMOTION_ADJ, FEATURES, feature_label

_component = None


def is_supported() -> bool:
    import streamlit as st
    return hasattr(getattr(st, "components", None), "v2") and hasattr(st.components.v2, "component")


def _get_component():
    global _component
    if _component is None:
        import streamlit as st
        _component = st.components.v2.component("song_mood_taste_mix", html=HTML, css=CSS, js=JS)
    return _component


def render(view: dict, key: str, on_feedback):
    """Mount the component. ``on_feedback`` runs before the rerun; read the click from
    ``st.session_state[key].feedback``."""
    return _get_component()(key=key, data=view, on_feedback_change=on_feedback)


# --- view model ---------------------------------------------------------------------------

_TRACK_FIELDS = (
    "key", "title", "artist", "rank", "base_rank", "personal_rank", "rank_change", "base_percent",
    "personal_percent", "preference", "taste_parts", "reasons", "state", "top_emotion",
    "secondary_emotion", "emotion_profile", "valence", "intensity", "why", "matched_keywords",
    "preview_url", "track_url", "duration_seconds", "category", "scores",
)


def _slim(track: dict) -> dict:
    out = {f: track.get(f) for f in _TRACK_FIELDS}
    themes = track.get("themes") or {}
    out["top_themes"] = [name for name, score in sorted(themes.items(), key=lambda kv: -kv[1]) if score >= 0.2][:3]
    return out


def _chart(history: list, top: int = 6) -> dict:
    if not history:
        return {"steps": [], "series": []}
    final = history[-1]["weights"]
    chosen = sorted(FEATURES, key=lambda n: -abs(final[n]))[:top]
    chosen = [n for n in chosen if abs(final[n]) > 1e-4]
    return {
        "steps": [{"step": h["step"], "action": h["action"], "title": h["title"], "artist": h["artist"],
                   "strength": h["strength"]} for h in history],
        "series": [{"feature": n, "label": feature_label(n), "values": [h["weights"][n] for h in history]}
                   for n in chosen],
    }


def build_view(result: dict, playlist: dict, history: list, last_change: list, saved: list,
               toast: dict | None, version: int, engine: dict) -> dict:
    """Everything the component draws, as plain JSON (no lyrics)."""
    profile = dict(result["profile"])
    profile.pop("raw_weights", None)
    return {
        "version": version,
        "query": playlist["query"],
        "k": result["k"],
        "database_size": playlist["database_size"],
        "alpha": result["alpha"],
        "engine": engine,
        "parsed": playlist["parsed_mood"],
        "weights": playlist["weights"],
        "personal": [_slim(t) for t in result["personal"]],
        "base": [_slim(t) for t in result["base"]],
        "deck": [_slim(t) for t in result["deck"]],
        "comparison": result["comparison"],
        "profile": profile,
        "emotion_labels": {e: EMOTION_ADJ[e] for e in EMOTIONS},
        "chart": _chart(history),
        "last_change": last_change,
        "saved": saved,
        "toast": toast,
        "playlist": playlist,
    }


def saved_items(states: dict, songs_by_key: dict) -> list:
    """Saved songs for the Saved tab. ``songs_by_key`` maps song key -> library record
    (for links / emotion); songs no longer in the database still show by title."""
    out = []
    for key, state in states.items():
        if not state.get("saved"):
            continue
        song = songs_by_key.get(key) or {}
        top = song.get("top_emotion")
        out.append({
            "key": key,
            "title": state.get("title") or song.get("title", ""),
            "artist": state.get("artist") or song.get("artist", ""),
            "top_emotion": top if top not in (None, "-") else None,
            "track_url": song.get("track_url") if str(song.get("track_url") or "").startswith("https://") else None,
        })
    return out


# --- front end ----------------------------------------------------------------------------

HTML = '<div class="smx" data-root="1"></div>'

CSS = r"""
:host { all: initial; }
.smx { --bg:#0f0c24; --bg2:#1b1340; --panel:rgba(255,255,255,.06); --panel2:rgba(255,255,255,.1);
  --line:rgba(255,255,255,.12); --text:#f5f3ff; --muted:#aaa6d6; --like:#2ee6a6; --nope:#ff5d73;
  --save:#ffc53d; --skip:#7aa2ff;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color:var(--text);
  background: radial-gradient(900px 420px at 0% 0%, #4b23a855, transparent 60%),
              radial-gradient(800px 420px at 100% 100%, #0e8f8a40, transparent 60%),
              linear-gradient(160deg, var(--bg), var(--bg2) 60%, #2a0f4a);
  border-radius:22px; padding:18px 18px 22px; position:relative; overflow:hidden; box-sizing:border-box; }
.smx *, .smx *::before, .smx *::after { box-sizing:border-box; }
.smx button { font:inherit; color:inherit; }
.smx button:focus-visible, .smx [tabindex]:focus-visible { outline:2px solid #fff; outline-offset:2px; }

/* header */
.top { display:flex; flex-wrap:wrap; gap:14px; align-items:center; justify-content:space-between; }
.ttl h2 { margin:0; font-size:22px; letter-spacing:.2px; }
.ttl .q { margin:3px 0 0; color:var(--muted); font-style:italic; font-size:14px; overflow-wrap:anywhere; }
.badges { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.badge { font-size:12px; padding:5px 10px; border-radius:999px; background:var(--panel); border:1px solid var(--line); color:var(--muted); }
.badge b { color:var(--text); }
.meter { display:flex; align-items:center; gap:10px; padding:6px 12px 6px 6px; border-radius:999px; background:var(--panel); border:1px solid var(--line); }
.meter .ring { --p:0; width:38px; height:38px; border-radius:50%; display:grid; place-items:center; font-size:11px; font-weight:700;
  background:conic-gradient(#f15bb5 calc(var(--p)*1%), rgba(255,255,255,.12) 0); position:relative; transition: --p .6s; }
.meter .ring::before { content:""; position:absolute; inset:4px; border-radius:50%; background:#1a1238; }
.meter .ring span { position:relative; }
.meter small { display:block; color:var(--muted); font-size:11px; }
.meter b { font-size:13px; }

/* tabs */
.tabs { display:flex; gap:6px; margin:16px 0 12px; overflow-x:auto; padding-bottom:2px; }
.tab { border:1px solid var(--line); background:var(--panel); padding:8px 14px; border-radius:999px; cursor:pointer; white-space:nowrap; font-size:14px; color:var(--muted); transition:background .15s, color .15s, transform .15s; }
.tab:hover { color:var(--text); transform:translateY(-1px); }
.tab[aria-selected=true] { color:#140b2e; background:linear-gradient(90deg,#ffd166,#f15bb5); border-color:transparent; font-weight:700; }
.tab .n { display:inline-block; min-width:18px; margin-left:6px; padding:0 6px; border-radius:999px; background:rgba(0,0,0,.25); font-size:11px; }
.panel { max-height:640px; overflow-y:auto; padding-right:4px; }
.panel::-webkit-scrollbar { width:9px; } .panel::-webkit-scrollbar-thumb { background:rgba(255,255,255,.2); border-radius:9px; }
.hint { color:var(--muted); font-size:13px; margin:0 0 12px; line-height:1.45; }

/* cards */
.card { position:relative; display:grid; grid-template-columns:auto 1fr auto; gap:6px 14px; align-items:center; padding:14px 14px 12px 20px;
  margin-bottom:10px; border-radius:18px; background:var(--panel); border:1px solid var(--line); overflow:hidden;
  transition: background .2s, box-shadow .2s, opacity .3s, transform .3s; }
.card:hover { background:var(--panel2); }
.card .bar { position:absolute; left:0; top:0; bottom:0; width:6px; }
.card.gone { opacity:0; transform:translateX(-40px) scale(.96); }
.card.skipped { opacity:.45; transform:translateX(30px); }
.card.liked { box-shadow: inset 0 0 0 1px #2ee6a688, 0 0 24px #2ee6a622; }
.card.saved { box-shadow: inset 0 0 0 1px #ffc53d99, 0 0 24px #ffc53d26; }
.rank { position:relative; width:44px; height:44px; border-radius:14px; display:grid; place-items:center; font-weight:800; font-size:18px; color:#fff; }
.delta { position:absolute; bottom:-8px; left:50%; transform:translateX(-50%); font-size:10px; font-weight:800; padding:1px 6px; border-radius:999px; white-space:nowrap; }
.delta.up { background:var(--like); color:#062b20; } .delta.down { background:var(--nope); color:#3a0610; } .delta.new { background:#ffd166; color:#3a2a00; }
.title { font-weight:700; font-size:16px; overflow-wrap:anywhere; cursor:pointer; }
.artist { color:var(--muted); font-size:13px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip { font-size:11.5px; padding:3px 9px; border-radius:999px; background:rgba(255,255,255,.09); border:1px solid var(--line); }
.chip.emo { color:#fff; border:0; font-weight:700; }
.chip.pos { border-color:#2ee6a666; color:#bff7e3; } .chip.neg { border-color:#ff5d7366; color:#ffd0d7; }
.score { text-align:right; min-width:86px; }
.score .big { font-size:22px; font-weight:800; }
.score small { display:block; color:var(--muted); font-size:11px; }
.acts { grid-column:1 / -1; display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:4px; }
.act { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:rgba(255,255,255,.07); padding:7px 12px; border-radius:999px; cursor:pointer; font-size:13px; transition:transform .12s, background .15s, border-color .15s; user-select:none; }
.act:hover { transform:translateY(-1px) scale(1.04); background:rgba(255,255,255,.14); }
.act:active { transform:scale(.94); }
.act .ic { font-size:15px; line-height:1; }
.act.like.on { background:#2ee6a633; border-color:var(--like); } .act.nope.on { background:#ff5d7333; border-color:var(--nope); }
.act.save.on { background:#ffc53d33; border-color:var(--save); } .act.skip:hover { border-color:var(--skip); }
.smx.pending .act, .smx.pending .bigbtn { pointer-events:none; opacity:.6; }
.more { margin-left:auto; color:var(--muted); font-size:12.5px; cursor:pointer; background:none; border:0; padding:6px 4px; }
.detail { grid-column:1 / -1; display:none; margin-top:8px; padding-top:12px; border-top:1px dashed var(--line); }
.card.open .detail { display:block; }
.why { margin:0 0 10px; font-size:14px; line-height:1.45; }
.bars { display:grid; gap:8px; margin:8px 0 12px; }
.brow { display:grid; grid-template-columns:150px 1fr 54px; gap:10px; align-items:center; font-size:12.5px; color:var(--muted); }
.track { height:8px; border-radius:8px; background:rgba(255,255,255,.12); overflow:hidden; position:relative; }
.fill { height:100%; border-radius:8px; }
.dtrack { position:relative; height:8px; border-radius:8px; background:rgba(255,255,255,.12); }
.dtrack::after { content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background:rgba(255,255,255,.4); }
.dfill { position:absolute; top:0; bottom:0; border-radius:8px; }
.stack { display:flex; height:12px; border-radius:8px; overflow:hidden; background:rgba(255,255,255,.1); margin-top:6px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 12px; margin:8px 0 10px; font-size:12px; color:var(--muted); }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.links { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.link { font-size:13px; padding:7px 12px; border-radius:10px; text-decoration:none; color:#fff; background:rgba(255,255,255,.12); }
.link:hover { background:rgba(255,255,255,.22); }
audio { height:34px; max-width:100%; }

/* bursts */
.burst { position:absolute; pointer-events:none; font-size:18px; animation: fly .9s ease-out forwards; z-index:5; }
@keyframes fly { 0% { transform:translate(0,0) scale(.6); opacity:1; } 100% { transform:translate(var(--dx), var(--dy)) scale(1.3) rotate(var(--rot)); opacity:0; } }
.pop { animation: pop .35s ease-out; } @keyframes pop { 0%{transform:scale(1);} 40%{transform:scale(1.35);} 100%{transform:scale(1);} }
.shake { animation: shake .3s; } @keyframes shake { 0%,100%{transform:translateX(0);} 25%{transform:translateX(-6px);} 75%{transform:translateX(6px);} }

/* deck */
.deckwrap { display:grid; place-items:center; padding:10px 0 4px; }
.deck { position:relative; width:min(380px, 92%); height:400px; }
.dcard { position:absolute; inset:0; border-radius:26px; padding:22px; color:#fff; display:flex; flex-direction:column; justify-content:flex-end;
  box-shadow:0 18px 40px rgba(0,0,0,.45); touch-action:none; user-select:none; transition: transform .35s cubic-bezier(.2,.8,.2,1), opacity .35s; overflow:hidden; background:#1a1238; }
.dcard.top { cursor:grab; } .dcard.dragging { transition:none; cursor:grabbing; }
.dcard .art { position:absolute; inset:0; }
.dcard .art::after { content:""; position:absolute; inset:0; background:linear-gradient(180deg, transparent 30%, rgba(8,4,24,.85)); }
.dcard .content { position:relative; }
.dcard h3 { margin:0; font-size:26px; line-height:1.1; overflow-wrap:anywhere; }
.dcard .by { margin:4px 0 10px; font-size:15px; opacity:.85; }
.dcard .note { font-size:12.5px; opacity:.85; margin-top:10px; }
.dcard .glyph { position:absolute; top:22px; right:22px; font-size:60px; opacity:.25; }
.stamp { position:absolute; top:28px; padding:6px 14px; border:4px solid; border-radius:12px; font-weight:900; font-size:26px; letter-spacing:2px; opacity:0; transform:rotate(-12deg); }
.stamp.like { left:22px; color:var(--like); border-color:var(--like); }
.stamp.nope { right:22px; color:var(--nope); border-color:var(--nope); transform:rotate(12deg); }
.stamp.save { left:50%; top:auto; bottom:150px; transform:translateX(-50%); color:var(--save); border-color:var(--save); }
.stamp.skip { left:50%; top:auto; bottom:150px; transform:translateX(-50%); color:var(--skip); border-color:var(--skip); }
.deckbtns { display:flex; gap:14px; justify-content:center; margin-top:16px; }
.bigbtn { width:62px; height:62px; border-radius:50%; border:2px solid var(--line); background:rgba(255,255,255,.06); font-size:26px; cursor:pointer; transition:transform .12s, background .15s; display:grid; place-items:center; }
.bigbtn:hover { transform:scale(1.1); background:rgba(255,255,255,.14); }
.bigbtn.like { border-color:var(--like); } .bigbtn.nope { border-color:var(--nope); } .bigbtn.save { border-color:var(--save); width:52px; height:52px; font-size:22px; align-self:center; } .bigbtn.skip { border-color:var(--skip); width:52px; height:52px; font-size:20px; align-self:center; }
.kbd { text-align:center; color:var(--muted); font-size:12px; margin-top:10px; }
.kbd kbd { background:rgba(255,255,255,.12); border-radius:6px; padding:1px 6px; font-family:inherit; }

/* taste */
.grid2 { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }
.box { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:14px; }
.box h4 { margin:0 0 10px; font-size:12px; letter-spacing:1px; text-transform:uppercase; color:var(--muted); }
.insight { font-size:16px; margin:0 0 8px; padding:10px 12px; border-radius:12px; background:linear-gradient(90deg, rgba(241,91,181,.18), rgba(122,162,255,.12)); }
.heat { display:grid; grid-template-columns: 96px repeat(3, 1fr); gap:4px; font-size:12px; }
.heat .h { color:var(--muted); text-align:center; padding:2px 0; }
.heat .rl { color:var(--text); display:flex; align-items:center; }
.heat .cell { border-radius:8px; padding:7px 0; text-align:center; font-weight:700; font-size:11.5px; transition:transform .15s; }
.heat .cell:hover { transform:scale(1.06); }
.trow { display:grid; grid-template-columns:118px 1fr 44px; gap:8px; align-items:center; font-size:12.5px; margin:5px 0; }
.trow .v { text-align:right; color:var(--muted); }
.change { display:flex; gap:8px; flex-wrap:wrap; }
.chg { font-size:12.5px; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.08); }
.chg.up { color:#bff7e3; } .chg.down { color:#ffd0d7; }
.chart svg, .slope svg { width:100%; height:auto; display:block; }
.tip { position:absolute; pointer-events:none; background:#0b0820; border:1px solid var(--line); border-radius:10px; padding:6px 9px; font-size:12px; white-space:nowrap; transform:translate(-50%, -110%); z-index:10; display:none; }

/* compare */
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-bottom:12px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px 12px; }
.stat b { display:block; font-size:22px; } .stat small { color:var(--muted); font-size:12px; }
.slope .lbl { cursor:pointer; }
.dropped { margin-top:10px; font-size:13px; color:var(--muted); }
.dropped li { margin:4px 0; }

/* saved */
.sgrid { display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:12px; }
.stile { border-radius:18px; padding:14px; min-height:120px; display:flex; flex-direction:column; justify-content:space-between; color:#fff; position:relative; overflow:hidden; }
.stile .t { font-weight:800; font-size:15px; overflow-wrap:anywhere; } .stile .a { font-size:12.5px; opacity:.85; }
.stile .row { display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; }
.mini { font-size:12px; padding:5px 9px; border-radius:999px; border:0; cursor:pointer; background:rgba(0,0,0,.28); color:#fff; text-decoration:none; }
.empty { text-align:center; padding:40px 10px; color:var(--muted); }
.empty .e { font-size:40px; display:block; margin-bottom:8px; }

/* json */
pre { margin:0; padding:14px; border-radius:14px; background:rgba(0,0,0,.35); border:1px solid var(--line); font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; overflow:auto; max-height:520px; color:#e9e9ff; }
.toolbar { display:flex; gap:8px; margin-bottom:10px; align-items:center; }

/* toast */
.toast { position:absolute; right:16px; bottom:16px; max-width:min(420px, 80%); background:#0b0820ee; border:1px solid var(--line); border-radius:14px; padding:10px 14px; font-size:13.5px; box-shadow:0 10px 30px rgba(0,0,0,.4); transform:translateY(20px); opacity:0; transition:all .3s; z-index:20; }
.toast.show { transform:translateY(0); opacity:1; }
.toast small { display:block; color:var(--muted); margin-top:3px; }
@media (max-width:560px) { .brow { grid-template-columns:96px 1fr 44px; } .score { min-width:64px; } .deck { height:360px; } }
"""

JS = r"""
const PALETTE = {
  joy:["#FFB800","#FF7A00"], trust:["#2EC4B6","#0E8F8A"], anticipation:["#FF9F1C","#F15B2A"],
  surprise:["#F15BB5","#9B5DE5"], fear:["#9B5DE5","#5B2A9D"], sadness:["#4F7CFF","#2B3FD6"],
  disgust:["#9BD43A","#4C9A2A"], anger:["#FF4D4D","#C2185B"], none:["#8892b0","#5b6486"]
};
const EMOTIONS = ["joy","trust","anticipation","surprise","fear","sadness","disgust","anger"];
const ICON = { like:"❤️", dislike:"👎", save:"⭐", skip:"⏭️", unsave:"☆", clear:"↺" };
const VERB = { like:"Liked", dislike:"Not for you", save:"Saved", skip:"Skipped", unsave:"Removed from saved", clear:"Cleared rating" };
const S = { tab:"foryou", open:null, version:null, pending:false, pendingTimer:null, lastToast:null, focus:null, keyHandler:null };

const pal = e => PALETTE[e] || PALETTE.none;
const grad = (e, deg) => `linear-gradient(${deg || 135}deg, ${pal(e)[0]}, ${pal(e)[1]})`;
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
const pct = x => Math.round(100 * x) + "%";
const isHttps = u => typeof u === "string" && u.startsWith("https://");
const fmtDur = s => (s == null ? null : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"));
const signed = (x, d) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d == null ? 2 : d);

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
const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, text) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

function burst(host, x, y, glyphs, n) {
  for (let i = 0; i < (n || 10); i++) {
    const a = Math.random() * Math.PI * 2, r = 50 + Math.random() * 70;
    const el = h("span", { class: "burst", style:
      `left:${x}px;top:${y}px;--dx:${Math.cos(a) * r}px;--dy:${Math.sin(a) * r - 30}px;--rot:${Math.random() * 360 - 180}deg` },
      glyphs[i % glyphs.length]);
    host.append(el);
    setTimeout(() => el.remove(), 950);
  }
}

export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  let root = parentElement.querySelector(".smx");
  if (!root) { root = h("div", { class: "smx" }); parentElement.append(root); }

  if (data.version !== S.version) {
    S.version = data.version; S.pending = false; clearTimeout(S.pendingTimer);
  }
  root.classList.toggle("pending", S.pending);

  const send = (action, key, delay) => {
    if (S.pending) return;
    S.pending = true; root.classList.add("pending");
    clearTimeout(S.pendingTimer);
    S.pendingTimer = setTimeout(() => { S.pending = false; root.classList.remove("pending"); }, 5000);
    const fire = () => setTriggerValue("feedback", { action, key, nonce: Date.now() + ":" + Math.random() });
    if (delay) setTimeout(fire, delay); else fire();
  };

  // FLIP: remember where each card was before re-rendering, animate to the new spot after.
  const before = new Map();
  root.querySelectorAll(".card[data-key]").forEach(el => before.set(el.dataset.key, el.getBoundingClientRect().top));

  root.textContent = "";
  root.append(header(data));
  const tabs = [
    ["foryou", "🎧 For you", data.personal.length],
    ["rate", "🃏 Rate", data.deck.length],
    ["taste", "🧬 Your taste", null],
    ["compare", "⚖️ Compare", null],
    ["saved", "⭐ Saved", data.saved.length],
    ["json", "{ } JSON", null],
  ];
  const tabbar = h("div", { class: "tabs", role: "tablist" });
  const panel = h("div", { class: "panel", role: "tabpanel" });
  const paint = () => {
    tabbar.textContent = "";
    for (const [id, label, n] of tabs) {
      tabbar.append(h("button", { class: "tab", role: "tab", "aria-selected": String(S.tab === id),
        onclick: () => { S.tab = id; paint(); } }, label, n != null ? h("span", { class: "n" }, n) : null));
    }
    panel.textContent = "";
    const views = { foryou: forYou, rate: rate, taste: taste, compare: compare, saved: saved, json: jsonView };
    panel.append(views[S.tab](data, send, root));
  };
  root.append(tabbar, panel);
  paint();

  requestAnimationFrame(() => {
    root.querySelectorAll(".card[data-key]").forEach(el => {
      const old = before.get(el.dataset.key);
      if (old === undefined) { el.animate([{ opacity: 0, transform: "translateY(12px)" }, { opacity: 1, transform: "none" }], { duration: 350, easing: "ease-out" }); return; }
      const dy = old - el.getBoundingClientRect().top;
      if (Math.abs(dy) > 2) el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 550, easing: "cubic-bezier(.2,.8,.2,1)" });
    });
  });

  showToast(root, data.toast);

  // keyboard shortcuts for the swipe deck
  if (S.keyHandler) window.removeEventListener("keydown", S.keyHandler);
  S.keyHandler = (ev) => {
    if (S.tab !== "rate" || !data.deck.length || S.pending) return;
    const t = ev.composedPath()[0];
    if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || "")) return;
    const map = { ArrowRight: "like", ArrowLeft: "dislike", ArrowUp: "save", ArrowDown: "skip" };
    const action = map[ev.key];
    if (!action || !root.isConnected) return;
    ev.preventDefault();
    const top = root.querySelector(".dcard.top");
    if (top) flyAway(top, action);
    send(action, data.deck[0].key, 260);
  };
  window.addEventListener("keydown", S.keyHandler);
  return () => { if (S.keyHandler) window.removeEventListener("keydown", S.keyHandler); };
}

// ---- header -------------------------------------------------------------------------------
function header(data) {
  const strength = data.profile.strength || 0;
  const eng = data.engine || {};
  const neural = (data.profile.engine || {}).neural_status;
  let engineText = "📐 Linear taste model";
  if (neural === "active") engineText = `🧠 Wide & Deep · ${Math.round(100 * (data.profile.engine.neural_share || 0))}% neural`;
  else if (eng.available) engineText = "🧠 Neural model warming up";
  return h("div", { class: "top" },
    h("div", { class: "ttl" }, h("h2", {}, "Your mood mix"), h("p", { class: "q" }, "“" + data.query + "”")),
    h("div", { class: "badges" },
      h("span", { class: "badge" }, h("b", {}, data.database_size), " songs searched"),
      h("span", { class: "badge", title: eng.label || "" }, engineText),
      h("div", { class: "meter", title: "How much your ratings shape the ranking. It grows as you rate more songs (max 45%)." },
        h("div", { class: "ring", style: `--p:${Math.round(100 * strength)}` }, h("span", {}, Math.round(100 * data.alpha) + "%")),
        h("div", {}, h("b", {}, "Personalization"), h("small", {}, `${data.profile.n_rated} rated · ${data.profile.n_events} actions`)))));
}

// ---- for you --------------------------------------------------------------------------------
function actionButtons(t, send, card) {
  const st = t.state || {};
  const btn = (cls, icon, label, on, action, burstGlyphs) => h("button", {
    class: `act ${cls}${on ? " on" : ""}`, title: label, "aria-pressed": String(!!on),
    onclick: (ev) => {
      ev.stopPropagation();
      const r = card.getBoundingClientRect(), b = ev.currentTarget.getBoundingClientRect();
      ev.currentTarget.classList.add("pop");
      if (burstGlyphs) burst(card, b.left - r.left + b.width / 2, b.top - r.top, burstGlyphs, action === "save" ? 14 : 9);
      if (action === "dislike") { card.classList.add("shake"); setTimeout(() => card.classList.add("gone"), 250); send(action, t.key, 380); return; }
      if (action === "skip") { card.classList.add("skipped"); send(action, t.key, 250); return; }
      send(action, t.key);
    } }, h("span", { class: "ic" }, icon), label);
  return [
    btn("like", "❤️", st.liked ? "Liked" : "Like", st.liked, st.liked ? "clear" : "like", st.liked ? null : ["❤️", "💖", "✨"]),
    btn("nope", "👎", "Nope", st.disliked, "dislike", null),
    btn("save", "⭐", st.saved ? "Saved" : "Save", st.saved, st.saved ? "unsave" : "save", st.saved ? null : ["⭐", "🌟", "✨", "🎉"]),
    btn("skip", "⏭️", st.skips ? `Skip (${st.skips})` : "Skip", false, "skip", null),
  ];
}

function deltaBadge(t) {
  if (t.rank_change == null || t.base_rank == null) return null;
  if (t.base_rank > 0 && t.rank_change === 0) return null;
  if (t.rank_change > 0) return h("span", { class: "delta up", title: `Was #${t.base_rank} before personalization` }, "▲" + t.rank_change);
  if (t.rank_change < 0) return h("span", { class: "delta down", title: `Was #${t.base_rank} before personalization` }, "▼" + (-t.rank_change));
  return null;
}

function tasteBar(value) {
  const v = Math.max(-1, Math.min(1, value || 0));
  const left = v >= 0 ? 50 : 50 + v * 50, width = Math.abs(v) * 50;
  return h("div", { class: "dtrack" }, h("div", { class: "dfill", style: `left:${left}%;width:${width}%;background:${v >= 0 ? "var(--like)" : "var(--nope)"}` }));
}

function trackCard(t, data, send, opts) {
  const emo = t.top_emotion;
  const card = h("div", { class: "card" + (t.state && t.state.liked ? " liked" : "") + (t.state && t.state.saved ? " saved" : "") + (S.open === t.key ? " open" : ""), "data-key": t.key });
  const chips = h("div", { class: "chips" });
  if (emo) chips.append(h("span", { class: "chip emo", style: `background:${grad(emo)}` }, cap(emo)));
  (t.top_themes || []).forEach(th => chips.append(h("span", { class: "chip" }, th)));
  (t.reasons || []).slice(0, 2).forEach(r => chips.append(h("span", { class: "chip " + (r.value >= 0 ? "pos" : "neg"), title: "How your taste profile moved this song" }, (r.value >= 0 ? "+ " : "− ") + r.text)));
  const dur = fmtDur(t.duration_seconds); if (dur) chips.append(h("span", { class: "chip" }, dur));
  const toggle = () => { S.open = S.open === t.key ? null : t.key; card.classList.toggle("open", S.open === t.key); };
  const pctShown = opts && opts.base ? t.base_percent : t.personal_percent;
  card.append(
    h("div", { class: "bar", style: `background:${grad(emo, 180)}` }),
    h("div", { class: "rank", style: `background:${grad(emo)}` }, t.rank, opts && opts.base ? null : deltaBadge(t)),
    h("div", {}, h("div", { class: "title", onclick: toggle }, t.title), h("div", { class: "artist" }, t.artist), chips),
    h("div", { class: "score" }, h("div", { class: "big" }, pctShown + "%"), h("small", {}, opts && opts.base ? "mood match" : "for you")),
    h("div", { class: "acts" }, actionButtons(t, send, card),
      h("button", { class: "more", onclick: toggle, "aria-expanded": String(S.open === t.key) }, S.open === t.key ? "less ▴" : "details ▾")),
    detail(t));
  return card;
}

function detail(t) {
  const d = h("div", { class: "detail" });
  d.append(h("p", { class: "why" }, t.why || ""));
  const parts = t.taste_parts || {};
  d.append(h("div", { class: "bars" },
    h("div", { class: "brow" }, h("span", {}, "Mood match"), h("div", { class: "track" }, h("div", { class: "fill", style: `width:${t.base_percent}%;background:linear-gradient(90deg,#4f7cff,#2ec4b6)` })), h("span", {}, t.base_percent + "%")),
    h("div", { class: "brow" }, h("span", {}, "Your taste"), tasteBar(t.preference), h("span", {}, signed(t.preference || 0))),
    parts.neural != null ? h("div", { class: "brow" }, h("span", {}, " ↳ linear / neural"), h("div", { class: "track", style: "background:none" }), h("span", {}, `${signed(parts.linear, 1)} / ${signed(parts.neural, 1)}`)) : null));
  const prof = t.emotion_profile || {};
  const stack = h("div", { class: "stack" });
  const entries = EMOTIONS.map(e => [e, prof[e] || 0]).filter(([, v]) => v > 0.001);
  entries.forEach(([e, v]) => stack.append(h("div", { title: `${cap(e)} ${pct(v)}`, style: `width:${v * 100}%;background:${pal(e)[0]}` })));
  d.append(h("div", { class: "hint", style: "margin:0" }, `Emotion mix · tone ${signed(t.valence || 0)} · energy ${signed(t.intensity || 0)}`), stack,
    h("div", { class: "legend" }, entries.slice().sort((a, b) => b[1] - a[1]).slice(0, 4).map(([e, v]) => h("span", {}, h("i", { class: "dot", style: `background:${pal(e)[0]}` }), `${cap(e)} ${pct(v)}`))));
  const q = encodeURIComponent(t.title + " " + t.artist);
  const links = h("div", { class: "links" });
  if (isHttps(t.preview_url)) links.append(h("audio", { controls: "", preload: "none", src: t.preview_url }));
  if (isHttps(t.track_url)) links.append(h("a", { class: "link", href: t.track_url, target: "_blank", rel: "noopener noreferrer" }, "Apple Music"));
  links.append(h("a", { class: "link", href: "https://open.spotify.com/search/" + q, target: "_blank", rel: "noopener noreferrer" }, "Spotify"));
  links.append(h("a", { class: "link", href: "https://www.youtube.com/results?search_query=" + q, target: "_blank", rel: "noopener noreferrer" }, "YouTube"));
  d.append(links);
  return d;
}

function forYou(data, send) {
  const wrap = h("div", {});
  const n = data.profile.n_rated;
  wrap.append(h("p", { class: "hint" }, n
    ? `Re-ranked with your taste (${Math.round(100 * data.alpha)}% weight). ▲/▼ show how far each song moved from the plain mood ranking.`
    : "This is the plain mood ranking. Like, nope, save or skip songs and watch the list re-rank itself around your taste."));
  data.personal.forEach(t => wrap.append(trackCard(t, data, send)));
  if (!data.personal.length) wrap.append(empty("🫥", "Nothing left: you've hidden every song for this mood. Clear a rating in Compare or try another description."));
  return wrap;
}

// ---- swipe deck --------------------------------------------------------------------------------
function flyAway(card, action) {
  const to = { like: "translate(140%, -10%) rotate(25deg)", dislike: "translate(-140%, -10%) rotate(-25deg)",
               save: "translate(0, -140%) scale(1.05)", skip: "translate(0, 120%) scale(.9)" }[action];
  card.style.transform = to; card.style.opacity = "0";
}

function rate(data, send) {
  const wrap = h("div", {});
  if (!data.deck.length) { wrap.append(empty("🏁", "You've rated every song that fits this mood. Try a new description to get a fresh deck.")); return wrap; }
  wrap.append(h("p", { class: "hint" }, "These are the songs your taste model is least sure about. Each swipe teaches it the most."));
  const deck = h("div", { class: "deck" });
  const shown = data.deck.slice(0, 3).reverse();
  shown.forEach((t, i) => {
    const isTop = i === shown.length - 1;
    const depth = shown.length - 1 - i;
    const emo = t.top_emotion;
    const stamps = ["like", "nope", "save", "skip"].map(k => h("div", { class: "stamp " + k }, { like: "LIKE", nope: "NOPE", save: "SAVE", skip: "SKIP" }[k]));
    const card = h("div", { class: "dcard" + (isTop ? " top" : ""), style: `transform: translateY(${depth * 12}px) scale(${1 - depth * 0.05}); z-index:${10 - depth}` },
      h("div", { class: "art", style: `background:${grad(emo, 160)}` }),
      h("div", { class: "glyph" }, emoGlyph(emo)),
      ...stamps,
      h("div", { class: "content" },
        h("h3", {}, t.title), h("div", { class: "by" }, t.artist),
        h("div", { class: "chips" }, emo ? h("span", { class: "chip", style: "background:rgba(0,0,0,.25);border:0" }, cap(emo)) : null,
          h("span", { class: "chip", style: "background:rgba(0,0,0,.25);border:0" }, describe(t)),
          ...(t.top_themes || []).slice(0, 2).map(th => h("span", { class: "chip", style: "background:rgba(0,0,0,.25);border:0" }, th))),
        h("div", { class: "note" }, `Mood match ${t.base_percent}% · your taste so far ${signed(t.preference || 0)}`)));
    if (isTop) dragify(card, stamps, (action) => send(action, t.key, 260));
    deck.append(card);
  });
  const top = data.deck[0];
  const act = (cls, glyph, action, label) => h("button", { class: "bigbtn " + cls, title: label, onclick: () => {
    const c = deck.querySelector(".dcard.top"); if (c) flyAway(c, action); send(action, top.key, 260); } }, glyph);
  wrap.append(h("div", { class: "deckwrap" }, deck,
    h("div", { class: "deckbtns" }, act("nope", "👎", "dislike", "Nope (←)"), act("skip", "⏭️", "skip", "Skip (↓)"), act("save", "⭐", "save", "Save (↑)"), act("like", "❤️", "like", "Like (→)"))),
    h("div", { class: "kbd" }, "Drag the card, or use ", h("kbd", {}, "←"), " nope · ", h("kbd", {}, "→"), " like · ", h("kbd", {}, "↑"), " save · ", h("kbd", {}, "↓"), " skip"),
    h("p", { class: "hint", style: "text-align:center;margin-top:6px" }, `${data.deck.length} song${data.deck.length === 1 ? "" : "s"} in the deck`));
  return wrap;
}

function dragify(card, stamps, onAction) {
  let sx = 0, sy = 0, dx = 0, dy = 0, active = false;
  const [like, nope, save, skip] = stamps;
  card.addEventListener("pointerdown", ev => { active = true; sx = ev.clientX; sy = ev.clientY; card.setPointerCapture(ev.pointerId); card.classList.add("dragging"); });
  card.addEventListener("pointermove", ev => {
    if (!active) return;
    dx = ev.clientX - sx; dy = ev.clientY - sy;
    card.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx / 14}deg)`;
    like.style.opacity = Math.max(0, Math.min(1, dx / 110));
    nope.style.opacity = Math.max(0, Math.min(1, -dx / 110));
    save.style.opacity = Math.abs(dx) < 60 ? Math.max(0, Math.min(1, -dy / 110)) : 0;
    skip.style.opacity = Math.abs(dx) < 60 ? Math.max(0, Math.min(1, dy / 110)) : 0;
  });
  const end = () => {
    if (!active) return;
    active = false; card.classList.remove("dragging");
    let action = null;
    if (dx > 110) action = "like"; else if (dx < -110) action = "dislike";
    else if (dy < -110) action = "save"; else if (dy > 110) action = "skip";
    if (action) { flyAway(card, action); onAction(action); }
    else { card.style.transform = ""; stamps.forEach(s => s.style.opacity = 0); }
    dx = dy = 0;
  };
  card.addEventListener("pointerup", end);
  card.addEventListener("pointercancel", end);
}

function emoGlyph(e) { return { joy: "☀️", trust: "🤝", anticipation: "✨", surprise: "⚡", fear: "🌘", sadness: "🌧️", disgust: "🥀", anger: "🔥" }[e] || "🎵"; }
function describe(t) {
  const tone = t.valence > 0.35 ? "bright" : t.valence < -0.35 ? "dark" : "mellow";
  const en = t.intensity > 0.5 ? "energetic" : t.intensity < -0.5 ? "calm" : "steady";
  return `${en}, ${tone}`;
}
function empty(e, text) { return h("div", { class: "empty" }, h("span", { class: "e" }, e), text); }

// ---- your taste -----------------------------------------------------------------------------------
function heatColor(v) {
  const a = Math.min(1, Math.abs(v) * 1.4);
  return v >= 0 ? `rgba(46,230,166,${0.12 + 0.75 * a})` : `rgba(255,93,115,${0.12 + 0.75 * a})`;
}

function taste(data, send, root) {
  const p = data.profile;
  const wrap = h("div", {});
  if (!p.n_rated) {
    wrap.append(empty("🧪", "Your taste profile is empty. Rate a few songs in For you or Rate and watch it build itself here."));
    return wrap;
  }
  const insights = h("div", { class: "box" }, h("h4", {}, "What I've learned"));
  (p.insights.length ? p.insights : ["Not enough of a pattern yet. Keep rating."]).forEach(t => insights.append(h("p", { class: "insight" }, t)));
  if (data.last_change && data.last_change.length) {
    const ch = h("div", { class: "change" });
    data.last_change.forEach(c => ch.append(h("span", { class: "chg " + (c.delta >= 0 ? "up" : "down") }, (c.delta >= 0 ? "↑ " : "↓ ") + c.label)));
    insights.append(h("h4", { style: "margin-top:12px" }, "Your last rating moved"), ch);
  }

  const heat = h("div", { class: "heat" }, h("span", {}), h("span", { class: "h" }, "calm"), h("span", { class: "h" }, "steady"), h("span", { class: "h" }, "energetic"));
  EMOTIONS.forEach(e => {
    const row = p.matrix[e];
    heat.append(h("span", { class: "rl" }, h("i", { class: "dot", style: `background:${pal(e)[0]}` }), data.emotion_labels[e]));
    ["calm", "steady", "energetic"].forEach(level => heat.append(h("span", { class: "cell", style: `background:${heatColor(row[level])}`,
      title: `Predicted taste for a ${level} ${data.emotion_labels[e]} song: ${signed(row[level])}` }, signed(row[level], 1))));
  });
  const heatBox = h("div", { class: "box" }, h("h4", {}, "Emotion × energy"), heat,
    h("p", { class: "hint", style: "margin:10px 0 0" }, "How much you'd like a typical song of each kind. Different colours across a row are the model telling “sad” apart from “sad and energetic”."));

  const themes = Object.entries(p.themes).sort((a, b) => b[1] - a[1]);
  const themeBox = h("div", { class: "box" }, h("h4", {}, "Lyrical themes"));
  themes.forEach(([name, v]) => themeBox.append(h("div", { class: "trow" }, h("span", {}, name), tasteBar(v), h("span", { class: "v" }, signed(v, 1)))));
  themeBox.append(h("div", { class: "trow", style: "margin-top:10px" }, h("span", {}, "bright tone"), tasteBar(p.tone), h("span", { class: "v" }, signed(p.tone, 1))),
    h("div", { class: "trow" }, h("span", {}, "high energy"), tasteBar(p.energy), h("span", { class: "v" }, signed(p.energy, 1))));

  const chartBox = h("div", { class: "box chart", style: "position:relative" }, h("h4", {}, "How your profile evolved"), evolution(data.chart));
  const eng = p.engine || {};
  const engBox = h("div", { class: "box" }, h("h4", {}, "Model"),
    h("p", { class: "hint", style: "margin:0" },
      eng.neural_status === "active"
        ? `Ranking blends the readable linear model with a Keras Wide & Deep network (${Math.round(100 * eng.neural_share)}% neural, grows with more ratings). Backend: ${eng.backend}.`
        : eng.neural_status === "not installed"
          ? "Linear taste model only. Install TensorFlow (or JAX) to add the Keras Wide & Deep network; see the README."
          : `Linear taste model for now; the Keras network kicks in after you've ${eng.neural_status ? eng.neural_status.replace("waiting for ", "rated ") : "rated a few songs"}.`));

  wrap.append(h("div", { class: "grid2" }, insights, heatBox), h("div", { style: "height:14px" }),
    h("div", { class: "grid2" }, chartBox, themeBox), h("div", { style: "height:14px" }), engBox);
  return wrap;
}

function evolution(chart) {
  const wrap = h("div", { style: "position:relative" });
  if (!chart.steps.length || !chart.series.length) { wrap.append(h("p", { class: "hint" }, "Rate a couple of songs to see this chart.")); return wrap; }
  const W = 560, H = 260, L = 34, R = 12, T = 12, B = 46;
  const n = chart.steps.length;
  const all = chart.series.flatMap(sr => sr.values).concat([0]);
  const lim = Math.max(0.05, ...all.map(Math.abs)) * 1.1;
  const X = i => L + (n === 1 ? (W - L - R) / 2 : i * (W - L - R) / (n - 1));
  const Y = v => T + (lim - v) / (2 * lim) * (H - T - B);
  const colors = ["#f15bb5", "#2ee6a6", "#ffd166", "#7aa2ff", "#ff9f1c", "#c58cff"];
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Preference weights over time" });
  svg.append(s("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "rgba(255,255,255,.35)", "stroke-dasharray": "4 4" }));
  svg.append(s("text", { x: 4, y: Y(lim * 0.9) + 4, fill: "#aaa6d6", "font-size": 10 }, "likes"));
  svg.append(s("text", { x: 4, y: Y(-lim * 0.9) + 4, fill: "#aaa6d6", "font-size": 10 }, "dislikes"));
  chart.series.forEach((sr, si) => {
    const d = sr.values.map((v, i) => (i ? "L" : "M") + X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" ");
    svg.append(s("path", { d, fill: "none", stroke: colors[si % colors.length], "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    svg.append(s("circle", { cx: X(n - 1), cy: Y(sr.values[n - 1]), r: 3.5, fill: colors[si % colors.length] }));
  });
  const ACT = { like: "❤️", dislike: "👎", save: "⭐", skip: "⏭️", unsave: "☆", clear: "↺" };
  const tip = h("div", { class: "tip" });
  chart.steps.forEach((st, i) => {
    svg.append(s("text", { x: X(i), y: H - B + 18, "font-size": 12, "text-anchor": "middle" }, ACT[st.action] || "•"));
    const hit = s("rect", { x: X(i) - 9, y: T, width: 18, height: H - T - B + 24, fill: "transparent", style: "cursor:pointer" });
    hit.addEventListener("pointerenter", () => {
      tip.textContent = `#${st.step} ${VERB[st.action] || st.action}: ${st.title}`;
      tip.style.display = "block"; tip.style.left = (X(i) / W * 100) + "%"; tip.style.top = "14px";
    });
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    svg.append(hit);
  });
  svg.append(s("text", { x: W / 2, y: H - 6, fill: "#aaa6d6", "font-size": 11, "text-anchor": "middle" }, "your last " + n + " actions →"));
  const legend = h("div", { class: "legend" }, chart.series.map((sr, si) => h("span", {}, h("i", { class: "dot", style: `background:${colors[si % colors.length]}` }), sr.label)));
  wrap.append(svg, tip, legend);
  return wrap;
}

// ---- compare --------------------------------------------------------------------------------------
function compare(data, send) {
  const c = data.comparison;
  const wrap = h("div", {});
  const stat = (v, label) => h("div", { class: "stat" }, h("b", {}, v), h("small", {}, label));
  wrap.append(h("div", { class: "stats" },
    stat(Math.round(100 * c.overlap) + "%", `of the original top ${data.k} kept`),
    stat(c.spearman == null ? "—" : c.spearman.toFixed(2), "rank correlation (Spearman)"),
    stat(c.moved, "songs moved"),
    stat(c.new_in_top.length, "new in your top " + data.k),
    stat(c.excluded, "hidden (you said nope)")));
  if (!data.profile.n_rated) wrap.append(h("p", { class: "hint" }, "No feedback yet, so both rankings are identical. Rate a few songs and come back."));

  const base = data.base, personal = data.personal;
  const W = 640, rowH = 34, top = 30, H = top + Math.max(base.length, personal.length, 1) * rowH + 40;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  const xL = 250, xR = 390;
  svg.append(s("text", { x: xL, y: 16, fill: "#aaa6d6", "font-size": 12, "text-anchor": "end" }, "Original (mood only)"));
  svg.append(s("text", { x: xR, y: 16, fill: "#aaa6d6", "font-size": 12 }, "Personalized"));
  const yOf = r => top + (r - 1) * rowH + rowH / 2;
  const outY = top + data.k * rowH + 18;
  const pRank = {}; personal.forEach(t => pRank[t.key] = t.rank);
  const bRank = {}; base.forEach(t => bRank[t.key] = t.rank);
  const lines = [];
  const add = (key, y1, y2, color, dash) => {
    const l = s("path", { d: `M${xL + 8},${y1} C${(xL + xR) / 2},${y1} ${(xL + xR) / 2},${y2} ${xR - 8},${y2}`, fill: "none", stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash || "", opacity: .85 });
    l.dataset.key = key; lines.push(l); svg.append(l);
  };
  base.forEach(t => {
    const pr = pRank[t.key];
    const color = pr == null ? "#ff5d73" : pr < t.rank ? "#2ee6a6" : pr > t.rank ? "#ff9f9f" : "rgba(255,255,255,.4)";
    add(t.key, yOf(t.rank), pr == null ? outY : yOf(pr), color, pr == null ? "4 4" : "");
  });
  personal.forEach(t => { if (bRank[t.key] == null) add(t.key, outY, yOf(t.rank), "#ffd166", "4 4"); });
  const label = (t, x, anchor, rank) => {
    const g = s("g", { class: "lbl" });
    g.append(s("text", { x, y: yOf(rank) + 4, fill: "#f5f3ff", "font-size": 12.5, "text-anchor": anchor },
      rank + ". " + trunc(t.title, 26)));
    g.append(s("circle", { cx: anchor === "end" ? xL + 4 : xR - 4, cy: yOf(rank), r: 4, fill: pal(t.top_emotion)[0] }));
    g.addEventListener("pointerenter", () => lines.forEach(l => l.setAttribute("opacity", l.dataset.key === t.key ? 1 : .15)));
    g.addEventListener("pointerleave", () => lines.forEach(l => l.setAttribute("opacity", .85)));
    svg.append(g);
  };
  base.forEach(t => label(t, xL, "end", t.rank));
  personal.forEach(t => label(t, xR, "start", t.rank));
  svg.append(s("text", { x: (xL + xR) / 2, y: outY + 4, fill: "#aaa6d6", "font-size": 11, "text-anchor": "middle" }, "below top " + data.k));
  wrap.append(h("div", { class: "box slope" }, h("h4", {}, "Original vs personalized"), svg,
    h("div", { class: "legend" }, h("span", {}, h("i", { class: "dot", style: "background:#2ee6a6" }), "moved up"), h("span", {}, h("i", { class: "dot", style: "background:#ff9f9f" }), "moved down"),
      h("span", {}, h("i", { class: "dot", style: "background:#ffd166" }), "new in top " + data.k), h("span", {}, h("i", { class: "dot", style: "background:#ff5d73" }), "dropped out"))));
  if (c.dropped.length) {
    const ul = h("ul", { class: "dropped" });
    c.dropped.forEach(d => {
      const disliked = d.why.indexOf("not for you") >= 0;
      ul.append(h("li", {}, `${d.title} (was #${d.base_rank}): ${d.why}. `,
        disliked ? h("button", { class: "mini", onclick: () => send("clear", d.key) }, "undo nope") : null));
    });
    wrap.append(h("div", { class: "box", style: "margin-top:12px" }, h("h4", {}, "Why songs dropped out"), ul));
  }
  return wrap;
}
const trunc = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);

// ---- saved -------------------------------------------------------------------------------------------
function saved(data, send) {
  const wrap = h("div", {});
  if (!data.saved.length) { wrap.append(empty("⭐", "Nothing saved yet. Hit Save on songs you want to keep; saves also count as your strongest signal.")); return wrap; }
  const grid = h("div", { class: "sgrid" });
  data.saved.forEach(t => {
    const q = encodeURIComponent(t.title + " " + t.artist);
    grid.append(h("div", { class: "stile", style: `background:${grad(t.top_emotion, 150)}` },
      h("div", {}, h("div", { class: "t" }, t.title), h("div", { class: "a" }, t.artist)),
      h("div", { class: "row" },
        isHttps(t.track_url) ? h("a", { class: "mini", href: t.track_url, target: "_blank", rel: "noopener noreferrer" }, "Apple Music") : null,
        h("a", { class: "mini", href: "https://open.spotify.com/search/" + q, target: "_blank", rel: "noopener noreferrer" }, "Spotify"),
        h("button", { class: "mini", onclick: () => send("unsave", t.key) }, "Remove"))));
  });
  wrap.append(h("div", { class: "toolbar" }, h("button", { class: "act", onclick: () => download("saved_songs.json", JSON.stringify(data.saved, null, 2)) }, "⬇️ Export saved songs")), grid);
  return wrap;
}

// ---- json ----------------------------------------------------------------------------------------------
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = h("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function jsonView(data) {
  const text = JSON.stringify(data.playlist, null, 2);
  const status = h("span", { class: "hint", style: "margin:0" }, "");
  return h("div", {},
    h("div", { class: "toolbar" },
      h("button", { class: "act", onclick: () => { (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => status.textContent = "Copied", () => status.textContent = "Copy blocked; select the text instead"); } }, "Copy JSON"),
      h("button", { class: "act", onclick: () => download("mood_playlist.json", text) }, "Download .json"), status),
    h("pre", {}, text));
}

// ---- toast ------------------------------------------------------------------------------------------------
function showToast(root, toast) {
  if (!toast || toast.id === S.lastToast) return;
  S.lastToast = toast.id;
  const el = h("div", { class: "toast", role: "status" }, toast.text, toast.detail ? h("small", {}, toast.detail) : null);
  root.append(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 400); }, 3800);
}
"""
