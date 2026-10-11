"""
Offline benchmark for the personalisation loop, using simulated listeners.

    python benchmark.py --token YOUR_GENIUS_TOKEN          # real lyrics (run on your machine)
    python benchmark.py --synthetic                        # smoke test on generated lyrics

What it measures
----------------
Each simulated listener has a hidden, hand-written taste (e.g. "likes sad songs only
when they're energetic"). For every listener x mood query x random seed:

1. Feedback is collected through the app's own loop: the listener rates the top card
   of the Rate deck (the song the model is least sure about), ``personalize()`` is
   re-run, and so on, up to each feedback budget.
2. At each budget, the *same* feedback is used to rank every song the listener hasn't
   rated yet, by: the plain mood ranking (no personalisation), the linear taste model,
   and the linear + Keras Wide & Deep blend (if Keras is installed). An ablation also
   ranks with the emotion x energy interaction terms switched off.
3. Rankings are scored with NDCG@K and precision@K against the listener's true
   relevance: ``fits the mood x fits their taste`` = mood similarity * sigmoid(2u).

Caveats (say "simulated listeners" if you quote these numbers): tastes are hand-written,
so this tests whether the loop recovers a known taste, not real user satisfaction.
The listeners' tastes are written over the same song features the model sees (with
noise and different functional forms), which favours the model; the "random taste"
listener is the control and should show no gain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager

import numpy as np

import neural_ranker
import preferences
import semantic
from emotion_engine import EMOTIONS
from mood_search import MoodIndex, song_key
from pipeline import CALIBRATION, get_lyrics, make_song_record
from preferences import FeedbackEvent, personalize
from themes import THEME_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SONGS = os.path.join(HERE, "benchmark_songs.txt")
DEFAULT_CACHE = os.path.join(HERE, "benchmark_cache.json")
QUERIES = [
    "chill but slightly nostalgic for a rainy day",
    "sad but energetic",
    "angry gym playlist",
    "very romantic, not sad",
    "dreamy late-night drive",
]
LIKE_Q, DISLIKE_Q, SAVE_Q = 0.70, 0.30, 0.90      # utility percentiles for like / dislike / save
FLIP_PROB = 0.10                                   # chance a like/dislike is recorded the wrong way round


# --- song set ------------------------------------------------------------------------------

def read_song_list(path: str) -> list:
    songs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            title, artist = (part.strip() for part in line.split("|", 1))
            songs.append((title, artist))
    return songs


def load_real_songs(token: str, list_path: str, cache_path: str) -> list:
    """Fetch lyrics once and cache them locally (the cache is gitignored: lyrics are copyrighted)."""
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    records, missing = [], []
    for title, artist in read_song_list(list_path):
        key = song_key(title, artist)
        if key not in cache:
            try:
                cache[key] = {"title": title, "artist": artist,
                              "lyrics": get_lyrics(title, artist, token, on_note=lambda m: print("  " + m))}
                print(f"  fetched {title} - {artist}")
                time.sleep(0.5)
            except Exception as e:      # network errors, missing songs
                missing.append(f"{title} - {artist} ({e})")
                continue
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
        entry = cache[key]
        records.append(make_song_record(entry["title"], entry["artist"], entry["lyrics"], with_track_info=False))
    if missing:
        print(f"Skipped {len(missing)} song(s):\n  " + "\n  ".join(missing))
    return records


def synthetic_songs(n: int, seed: int = 0) -> list:
    """Songs stitched from the test fixtures' lines - only for checking the script runs."""
    from tests.fixtures import SONGS
    rng = random.Random(seed)
    pools = [[l for l in lyrics.strip().split("\n") if l.strip() and not l.startswith("[")] for _, lyrics in SONGS.values()]
    records = []
    for i in range(n):
        picked = rng.sample(pools, 2)
        weight = rng.random()
        lines = [rng.choice(picked[0] if rng.random() < weight else picked[1]) for _ in range(14)]
        records.append(make_song_record(f"Synthetic {i + 1}", "Benchmark", "\n".join(lines), with_track_info=False))
    return records


# --- simulated listeners ---------------------------------------------------------------------

def feature_table(index: MoodIndex) -> dict:
    """Standardised song features the listeners' tastes are written in."""
    cols = {"tone": [], "energy": []}
    cols.update({e: [] for e in EMOTIONS})
    cols.update({t: [] for t in THEME_NAMES})
    for s in index.songs:
        cols["tone"].append(s["valence"])
        cols["energy"].append(s["intensity"])
        for i, e in enumerate(EMOTIONS):
            cols[e].append(float(s["profile"][i]))
        for t in THEME_NAMES:
            cols[t].append(float(s["themes"].get(t, 0.0)))
    z = {}
    for name, values in cols.items():
        arr = np.array(values, dtype=float)
        sd = arr.std()
        z[name] = (arr - arr.mean()) / sd if sd > 1e-9 else np.zeros_like(arr)
    return z


LISTENERS = {
    "likes sad songs": lambda z: z["sadness"],
    "likes sad songs only when energetic": lambda z: z["sadness"] * np.tanh(1.5 * z["energy"]),
    "likes bright, calm songs": lambda z: z["tone"] - z["energy"],
    "likes lyrics about memories and rain": lambda z: z["memories"] + z["rain & weather"],
    "likes joy, can't stand anger": lambda z: z["joy"] - 1.5 * np.maximum(0.0, z["anger"]),
    "random taste (control)": lambda z: np.zeros_like(z["tone"]),
}


def listener_utility(name: str, z: dict, rng: np.random.Generator) -> np.ndarray:
    base = LISTENERS[name](z)
    noise_scale = 1.0 if name.startswith("random") else 0.3
    return base + noise_scale * rng.standard_normal(len(base))


# --- metrics -----------------------------------------------------------------------------------

def ndcg_at_k(order: list, relevance: dict, k: int) -> float:
    gains = [relevance[key] for key in order[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(order: list, liked: set, k: int) -> float:
    top = order[:k]
    return sum(1 for key in top if key in liked) / max(1, len(top))


@contextmanager
def neural(enabled: bool):
    original = neural_ranker.available
    if not enabled:
        neural_ranker.available = lambda: False
    try:
        yield
    finally:
        neural_ranker.available = original


@contextmanager
def interactions_off():
    original = preferences._PENALTY_VECTOR.copy()
    mask = np.array(["*" in name for name in preferences.FEATURES])
    preferences._PENALTY_VECTOR[mask] = 1e6          # weights forced to ~0
    try:
        yield
    finally:
        preferences._PENALTY_VECTOR[:] = original


# --- one run -----------------------------------------------------------------------------------

def run_one(index, sq, utility, budgets, k, use_neural, seed):
    keys = [s["key"] for s in index.songs]
    u = dict(zip(keys, utility))
    like_t, dislike_t, save_t = (float(np.quantile(utility, q)) for q in (LIKE_Q, DISLIKE_Q, SAVE_Q))
    sim = dict(zip(keys, (float(x) for x in sq.similarity)))
    rng = random.Random(seed)

    events, rated, results = [], set(), {}
    for step in range(1, max(budgets) + 1):
        with neural(False):                              # collect feedback with the fast linear loop
            deck = personalize(index, sq, events, k=k)["deck"]
        if not deck:
            break
        track = deck[0]
        value = u[track["key"]]
        if value >= like_t:
            action = "like"
        elif value <= dislike_t:
            action = "dislike"
        else:
            action = "skip"
        if action in ("like", "dislike") and rng.random() < FLIP_PROB:
            action = "dislike" if action == "like" else "like"
        events.append(FeedbackEvent.create(action, track))
        if action == "like" and value >= save_t:
            events.append(FeedbackEvent.create("save", track))
        rated.add(track["key"])

        if step in budgets:
            candidates = [key for key in keys if key not in rated]
            relevance = {key: sim[key] / (1 + math.exp(-2 * u[key])) for key in candidates}
            liked = {key for key in candidates if u[key] >= like_t}
            only = lambda ranking: [key for key in ranking if key in relevance]
            orders = {"mood only (no personalization)": only(sorted(candidates, key=lambda key: -sim[key]))}
            with neural(False):
                orders["linear taste model"] = only(personalize(index, sq, events, k=k)["ranking"])
                with interactions_off():
                    orders["linear, no interaction terms"] = only(personalize(index, sq, events, k=k)["ranking"])
            if use_neural:
                with neural(True):
                    orders["linear + Wide & Deep"] = only(personalize(index, sq, events, k=k)["ranking"])
            results[step] = {name: {"ndcg": ndcg_at_k(order, relevance, k), "precision": precision_at_k(order, liked, k)}
                             for name, order in orders.items()}
    return results


# --- report ------------------------------------------------------------------------------------

def summarise(rows: list) -> dict:
    """rows: list of (listener, budget, system, metric, value) -> nested means and 95% CIs."""
    out = {}
    for listener, budget, system, metric, value in rows:
        out.setdefault(budget, {}).setdefault(system, {}).setdefault(metric, {}).setdefault(listener, []).append(value)
    return out


def mean_ci(values: list) -> tuple:
    m = statistics.fmean(values)
    ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, ci


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--token", help="Genius API token (fetches and caches real lyrics)")
    src.add_argument("--synthetic", action="store_true", help="smoke test on generated lyrics (numbers not meaningful)")
    parser.add_argument("--songs", default=DEFAULT_SONGS, help="song list file, 'Title | Artist' per line")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="local lyrics cache (gitignored)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 25], help="feedback counts to evaluate at")
    parser.add_argument("--seeds", type=int, default=3, help="random seeds per listener and query")
    parser.add_argument("--k", type=int, default=5, help="cutoff for NDCG@K / precision@K")
    parser.add_argument("--neural", choices=["auto", "on", "off"], default="auto",
                        help="include the Keras Wide & Deep blend (auto = if installed)")
    parser.add_argument("--semantic", choices=["auto", "off"], default="auto",
                        help="use transformer semantic search in the base ranking (auto = if installed)")
    parser.add_argument("--out", default=os.path.join(HERE, "benchmark_results.json"))
    args = parser.parse_args(argv)

    if args.semantic == "off":
        semantic.set_encoder(False)
    use_neural = args.neural == "on" or (args.neural == "auto" and neural_ranker.available())
    if args.neural == "on" and not neural_ranker.available():
        sys.exit("--neural on, but Keras with a TensorFlow/JAX backend isn't installed.")

    print("Loading songs...")
    songs = synthetic_songs(60) if args.synthetic else load_real_songs(args.token, args.songs, args.cache)
    budgets = sorted(set(b for b in args.budgets if b > 0))
    if len(songs) < max(budgets) + 2 * args.k:
        sys.exit(f"Need at least {max(budgets) + 2 * args.k} songs for these budgets; got {len(songs)}.")
    index = MoodIndex(songs, CALIBRATION)
    z = feature_table(index)
    scored = {q: index.score_all(q) for q in QUERIES}
    print(f"{len(songs)} songs, {len(LISTENERS)} listeners, {len(QUERIES)} queries, {args.seeds} seeds, "
          f"budgets {budgets}, lyric search={'semantic' if index.semantic is not None else 'keywords'}, "
          f"neural={'on' if use_neural else 'off'}"
          + (f" ({neural_ranker.backend_info()['label']})" if use_neural else ""))

    rows, started = [], time.time()
    for li, listener in enumerate(LISTENERS):
        for seed in range(args.seeds):
            utility = listener_utility(listener, z, np.random.default_rng(1000 * li + seed))
            for qi, query in enumerate(QUERIES):
                res = run_one(index, scored[query], utility, budgets, args.k, use_neural, seed=10_000 * li + 100 * seed + qi)
                for budget, systems in res.items():
                    for system, metrics in systems.items():
                        for metric, value in metrics.items():
                            rows.append((listener, budget, system, metric, value))
        print(f"  done: {listener}  ({time.time() - started:.0f}s)")

    table = summarise(rows)
    report = {"songs": len(songs), "synthetic": args.synthetic, "k": args.k, "seeds": args.seeds,
              "lyric_search": "semantic" if index.semantic is not None else "keywords",
              "queries": QUERIES, "listeners": list(LISTENERS), "neural": use_neural, "results": {}}
    real = [l for l in LISTENERS if not l.startswith("random")]

    print(f"\nNDCG@{args.k} and precision@{args.k} on songs each listener hadn't rated "
          f"(mean ± 95% CI over listeners x queries x seeds; control listener excluded)")
    for budget in budgets:
        print(f"\nAfter {budget} ratings:")
        print(f"  {'system':34s} {'NDCG@'+str(args.k):>16s} {'precision@'+str(args.k):>16s}")
        report["results"][budget] = {}
        for system, metrics in table[budget].items():
            nd = [v for l in real for v in metrics["ndcg"][l]]
            pr = [v for l in real for v in metrics["precision"][l]]
            (nm, nci), (pm, pci) = mean_ci(nd), mean_ci(pr)
            print(f"  {system:34s} {nm:8.3f} ± {nci:.3f} {pm:8.3f} ± {pci:.3f}")
            report["results"][budget][system] = {
                "ndcg": round(nm, 4), "ndcg_ci95": round(nci, 4), "precision": round(pm, 4), "precision_ci95": round(pci, 4),
                "per_listener_ndcg": {l: round(statistics.fmean(metrics["ndcg"][l]), 4) for l in LISTENERS},
            }

    top = max(budgets)
    print(f"\nPer listener, NDCG@{args.k} after {top} ratings:")
    systems = list(table[top])
    print("  " + f"{'listener':40s}" + "".join(f"{s[:22]:>24s}" for s in systems))
    for listener in LISTENERS:
        print("  " + f"{listener:40s}" + "".join(f"{statistics.fmean(table[top][s]['ndcg'][listener]):24.3f}" for s in systems))

    baseline = "mood only (no personalization)"
    base = report["results"][top][baseline]["ndcg"]
    report["headline"] = {"budget": top, "baseline_ndcg": base, "listeners": len(real), "systems": {}}
    print(f"\nHeadline after {top} ratings ({len(real)} simulated listeners, NDCG@{args.k} vs {base:.3f} without personalization):")
    for system in report["results"][top]:
        if system in (baseline, "linear, no interaction terms"):
            continue
        value = report["results"][top][system]["ndcg"]
        lift = 100 * (value - base) / base if base else float("nan")
        report["headline"]["systems"][system] = {"ndcg": value, "relative_lift_pct": round(lift, 1)}
        print(f"  {system}: {value:.3f} ({lift:+.1f}%)")
    control = report["results"][top]
    print(f"  control listener (random taste): {control['linear taste model']['per_listener_ndcg']['random taste (control)']:.3f} "
          f"personalized vs {control[baseline]['per_listener_ndcg']['random taste (control)']:.3f} without - "
          "personalization can't help (and can hurt) when there's no consistent taste to learn.")
    if args.synthetic:
        print("(Synthetic lyrics: this run only checks the script works. Don't quote these numbers.)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved {args.out}")
    return report


if __name__ == "__main__":
    main()
