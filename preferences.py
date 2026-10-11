"""
Personal taste model: learn from like / dislike / save / skip and re-rank mood searches.

Every song is described by a feature vector in standardised units ("z-scores"
relative to the songs in the current database):

    tone                 VADER mean, darker <-> brighter
    energy               VADER variance (line-to-line emotional swing), calm <-> energetic
    emo:<emotion>        share of each of the 8 NRC emotions in the lyrics
    theme:<theme>        how much the lyrics dwell on each theme in themes.py
    emo:<emotion>*energy interaction terms: emotion x energy
    tone*energy          interaction term: tone x energy

The interaction terms are what separate "I like sad songs" (a positive weight on
``emo:sadness``) from "I like sad songs *that are energetic*" (a positive weight on
``emo:sadness*energy``, possibly with no main-effect preference for sadness at all).

Fitting is weighted ridge regression on the user's effective labels, solved in
closed form: ``w = (X'CX + L)^-1 X'Cy``. Ridge is exact, order-independent and
cheap at this size (~30 features), so the whole profile can be refit after
every click and replayed to show how it evolved. Interaction terms carry a
three-times-larger penalty than main effects, so the model reaches for "sad
*and* energetic" only when the ratings actually need it.

Labels: like = +1, dislike = -1, save = +1 (and slightly stronger than a like),
skip = a weak -1 (a skip says "not now" more than "never"). The latest explicit
rating of a song wins, so changing your mind replaces the old rating.

When Keras is installed, a small Wide & Deep network (``neural_ranker.py``) is
trained on the same ratings and blended in:

    taste = (1 - b) * tanh(w . x) + b * neural(x),   b = 0.6 * n / (n + 10)

so the interpretable ridge model dominates until there are enough ratings for
the network to be worth listening to. The ridge weights are always what the
"Your taste" view explains, because they can be read directly.

Re-ranking blends the mood-match score with the learned taste:

    personal = (1 - a) * mood_match + a * (taste + 1) / 2  -  0.04 * skips

where the personalisation weight ``a`` grows with the amount of feedback
(0 with none, approaching 0.45). Disliked songs are removed from personal results.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass

import numpy as np

import neural_ranker
from emotion_engine import EMOTIONS
from themes import THEME_NAMES

ACTIONS = ("like", "dislike", "save", "unsave", "skip", "clear")

MAIN_FEATURES = ["tone", "energy"] + [f"emo:{e}" for e in EMOTIONS] + [f"theme:{t}" for t in THEME_NAMES]
INTERACTION_FEATURES = [f"emo:{e}*energy" for e in EMOTIONS] + ["tone*energy"]
FEATURES = MAIN_FEATURES + INTERACTION_FEATURES
_INDEX = {name: i for i, name in enumerate(FEATURES)}

# Ridge penalty per feature group (in z units). Higher = needs more evidence.
PENALTY = {"tone": 1.0, "energy": 1.0, "emo": 1.0, "theme": 1.5, "interaction": 3.0}
_STD_FLOOR = {"tone": 0.35, "energy": 0.35, "emo": 0.04, "theme": 0.05}
_DEFAULT_MEAN = {"tone": 0.0, "energy": 0.0, "emo": 1.0 / len(EMOTIONS), "theme": 0.15}

ALPHA_MAX = 0.45        # maximum weight of the learned preference in the final score
ALPHA_HALF = 4.0        # amount of (weighted) feedback at which half of ALPHA_MAX is reached
SKIP_PENALTY = 0.04     # per skip, up to 3, applied directly to the personal score
MAX_STORED_EVENTS = 500
NEURAL_MAX = 0.6        # maximum share of the neural model in the taste score
NEURAL_HALF = 10.0      # rated songs at which the neural share reaches half of NEURAL_MAX
HISTORY_WINDOW = 40

EMOTION_ADJ = {
    "joy": "joyful", "trust": "warm", "anticipation": "hopeful", "surprise": "surprising",
    "fear": "anxious", "sadness": "sad", "disgust": "bitter", "anger": "angry",
}
_EMOTION_TONE = {   # typical tone of a song dominated by each emotion (z units)
    "joy": 0.9, "trust": 0.6, "anticipation": 0.3, "surprise": 0.0,
    "fear": -0.6, "sadness": -0.9, "disgust": -0.6, "anger": -0.7,
}
ENERGY_LEVELS = (("calm", -1.2), ("steady", 0.0), ("energetic", 1.2))


def _group(name: str) -> str:
    if "*" in name:
        return "interaction"
    return name.split(":")[0]


# --- Feature space ---------------------------------------------------------------------

def raw_features(track: dict) -> dict:
    """The raw (unstandardised) features of a track dict from MoodIndex.track()."""
    return {
        "tone": float(track.get("valence", 0.0)),
        "energy": float(track.get("intensity", 0.0)),
        "emotions": {e: float((track.get("emotion_profile") or {}).get(e, 0.0)) for e in EMOTIONS},
        "themes": {t: float((track.get("themes") or {}).get(t, 0.0)) for t in THEME_NAMES},
    }


def _main_vector(raw: dict) -> np.ndarray:
    vals = [raw.get("tone", 0.0), raw.get("energy", 0.0)]
    vals += [raw.get("emotions", {}).get(e, 0.0) for e in EMOTIONS]
    vals += [raw.get("themes", {}).get(t, 0.0) for t in THEME_NAMES]
    return np.array(vals, dtype=float)


class FeatureSpace:
    """Standardises raw features against a reference population of songs and adds
    the interaction terms."""

    def __init__(self, population: list):
        groups = [_group(n) for n in MAIN_FEATURES]
        defaults = np.array([_DEFAULT_MEAN[g] for g in groups])
        floors = np.array([_STD_FLOOR[g] for g in groups])
        if len(population) >= 3:
            mat = np.array([_main_vector(r) for r in population])
            self.mean = mat.mean(axis=0)
            self.std = np.maximum(mat.std(axis=0), floors)
        else:
            self.mean, self.std = defaults, floors

    def z_main(self, raw: dict) -> np.ndarray:
        return (_main_vector(raw) - self.mean) / self.std

    def from_z(self, z_main: np.ndarray) -> np.ndarray:
        """Full feature vector (main effects + interactions) from standardised main effects."""
        z_tone, z_energy = z_main[0], z_main[1]
        z_emo = z_main[2:2 + len(EMOTIONS)]
        inter = list(z_emo * z_energy) + [z_tone * z_energy]
        return np.concatenate([z_main, np.array(inter)])

    def transform(self, raw: dict) -> np.ndarray:
        return self.from_z(self.z_main(raw))


# --- Feedback events -------------------------------------------------------------------

@dataclass
class FeedbackEvent:
    action: str
    key: str
    title: str
    artist: str
    features: dict          # raw features of the song when it was rated
    query: str = ""
    ts: float = 0.0
    id: str = ""

    @classmethod
    def create(cls, action: str, track: dict, query: str = "", event_id: str | None = None):
        if action not in ACTIONS:
            raise ValueError(f"Unknown feedback action {action!r}")
        return cls(action=action, key=track["key"], title=track["title"], artist=track["artist"],
                   features=raw_features(track), query=query, ts=time.time(),
                   id=event_id or uuid.uuid4().hex)

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEvent":
        return cls(**{k: d.get(k) for k in ("action", "key", "title", "artist", "features", "query", "ts", "id")})


def song_states(events: list) -> dict:
    """key -> {liked, disliked, saved, skips, title, artist, features} after replaying events."""
    states = {}
    for ev in events:
        st = states.setdefault(ev.key, {"liked": False, "disliked": False, "saved": False, "skips": 0})
        st.update(title=ev.title, artist=ev.artist, features=ev.features)
        if ev.action == "like":
            st["liked"], st["disliked"] = True, False
        elif ev.action == "dislike":
            st["liked"], st["disliked"] = False, True
        elif ev.action == "clear":
            st["liked"] = st["disliked"] = False
        elif ev.action == "save":
            st["saved"] = True
        elif ev.action == "unsave":
            st["saved"] = False
        elif ev.action == "skip":
            st["skips"] += 1
    return states


def label_for(state: dict):
    """(target, weight) for a song state, or None if it carries no signal."""
    if state["liked"]:
        return 1.0, 1.5 if state["saved"] else 1.0
    if state["disliked"]:
        return -1.0, 1.0
    if state["saved"]:
        return 1.0, 1.25
    if state["skips"]:
        return -1.0, 0.3 * min(state["skips"], 3)
    return None


# --- Model -----------------------------------------------------------------------------

_PENALTY_VECTOR = np.array([PENALTY[_group(n)] for n in FEATURES])


def fit_ridge(X: np.ndarray, y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Weighted ridge regression in closed form: w = (X'CX + L)^-1 X'Cy."""
    A = X.T @ (c[:, None] * X) + np.diag(_PENALTY_VECTOR)
    return np.linalg.solve(A, X.T @ (c * y))


class PreferenceModel:
    """Ridge taste model, optionally blended with the Keras Wide & Deep network.

    ``neural=True`` trains (or reuses from cache) the network when Keras is installed and
    there are enough ratings; profile-history replays leave it off for speed."""

    def __init__(self, events: list, space: FeatureSpace, neural: bool = False):
        self.events = list(events)
        self.space = space
        self.states = song_states(self.events)
        rows, deep_rows, targets, weights = [], [], [], []
        for state in self.states.values():
            label = label_for(state)
            if label is None:
                continue
            z = space.z_main(state["features"])
            rows.append(space.from_z(z))
            deep_rows.append(z)
            targets.append(label[0])
            weights.append(label[1])
        self.n_labeled = len(rows)
        self.n_eff = float(sum(weights))
        self.X = np.array(rows).reshape(-1, len(FEATURES))
        self.Z = np.array(deep_rows).reshape(-1, len(MAIN_FEATURES))
        self.y = np.array(targets, dtype=float)
        self.c = np.array(weights, dtype=float)
        self.w = fit_ridge(self.X, self.y, self.c) if rows else np.zeros(len(FEATURES))

        self.neural = None
        self.neural_status = "off"
        if neural:
            if not neural_ranker.available():
                self.neural_status = "not installed"
            else:
                self.neural = neural_ranker.trained_model(self.X, self.Z, self.y, self.c)
                self.neural_status = ("active" if self.neural is not None else
                                      f"waiting for {neural_ranker.MIN_RATINGS}+ rated songs incl. a like and a dislike/skip")

    @property
    def alpha(self) -> float:
        return ALPHA_MAX * self.n_eff / (self.n_eff + ALPHA_HALF)

    @property
    def beta(self) -> float:
        """Share of the neural model in the taste score (0 when it isn't active)."""
        if self.neural is None:
            return 0.0
        return NEURAL_MAX * self.n_labeled / (self.n_labeled + NEURAL_HALF)

    def weight(self, name: str) -> float:
        return float(self.w[_INDEX[name]])

    def score_vector(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    def preference(self, raw: dict) -> float:
        """Learned preference for a song in (-1, 1); 0 means no opinion."""
        return float(self.preferences([raw])["taste"][0])

    def preferences(self, raws: list) -> dict:
        """Batch scoring: {'taste', 'linear', 'neural'} arrays in (-1, 1) ('neural' may be None)."""
        if not raws:
            empty = np.zeros(0)
            return {"taste": empty, "linear": empty, "neural": None}
        Z = np.array([self.space.z_main(r) for r in raws])
        X = np.array([self.space.from_z(z) for z in Z])
        linear = np.tanh(X @ self.w)
        if self.neural is None:
            return {"taste": linear, "linear": linear, "neural": None}
        neural = self.neural.predict(X, Z)
        b = self.beta
        return {"taste": (1 - b) * linear + b * neural, "linear": linear, "neural": neural}

    def check(self) -> dict | None:
        """K-fold comparison of linear vs neural accuracy on this user's own ratings."""
        return neural_ranker.cross_validate(self.X, self.Z, self.y, self.c, fit_ridge)

    # -- explanation ------------------------------------------------------------------
    def contributions(self, raw: dict, top: int = 3) -> list:
        """The features that moved this song up or down the most, in plain words."""
        z = self.space.z_main(raw)
        x = self.space.from_z(z)
        contrib = self.w * x
        out = []
        for i in np.argsort(-np.abs(contrib)):
            if abs(contrib[i]) < 0.03 or len(out) >= top:
                break
            out.append({"text": feature_phrase(FEATURES[i], z), "value": round(float(contrib[i]), 3)})
        return out

    def emotion_energy_matrix(self) -> dict:
        """Predicted preference for a typical song dominated by each emotion, at three energy
        levels. This is where 'sad' vs 'sad and energetic' becomes visible."""
        out = {}
        n_emo = len(EMOTIONS)
        for e_i, e in enumerate(EMOTIONS):
            row = {}
            for label, energy in ENERGY_LEVELS:
                z = np.zeros(len(MAIN_FEATURES))
                z[0] = _EMOTION_TONE[e]
                z[1] = energy
                z[2:2 + n_emo] = -1.5 / (n_emo - 1)
                z[2 + e_i] = 1.5
                row[label] = round(math.tanh(self.score_vector(self.space.from_z(z))), 3)
            out[e] = row
        return out

    def summary(self) -> dict:
        scale = float(np.max(np.abs(self.w))) or 1.0
        norm = lambda name: round(self.weight(name) / scale, 3)
        matrix = self.emotion_energy_matrix()
        return {
            "engine": {
                "neural_status": self.neural_status,
                "neural_share": round(self.beta, 3),
                "backend": neural_ranker.backend_info()["label"] if self.neural_status != "off" else None,
            },
            "n_events": len(self.events),
            "n_rated": self.n_labeled,
            "evidence": round(self.n_eff, 2),
            "strength": round(self.alpha / ALPHA_MAX, 3),
            "alpha": round(self.alpha, 3),
            "tone": norm("tone"),
            "energy": norm("energy"),
            "emotions": {e: norm(f"emo:{e}") for e in EMOTIONS},
            "themes": {t: norm(f"theme:{t}") for t in THEME_NAMES},
            "interactions": {e: norm(f"emo:{e}*energy") for e in EMOTIONS},
            "tone_energy": norm("tone*energy"),
            "matrix": matrix,
            "insights": self.insights(matrix),
            "raw_weights": {n: round(self.weight(n), 4) for n in FEATURES},
        }

    def insights(self, matrix: dict | None = None) -> list:
        """Up to six plain-language statements about the learned taste."""
        if self.n_labeled == 0:
            return []
        matrix = matrix or self.emotion_energy_matrix()
        found = []   # (strength, text)
        for e, row in matrix.items():
            calm, steady, energetic = row["calm"], row["steady"], row["energetic"]
            adj = EMOTION_ADJ[e]
            peak = max(abs(calm), abs(steady), abs(energetic))
            if peak < 0.12:
                continue
            gap = energetic - calm
            if gap > 0.2 and energetic > 0.12:
                text = (f"You like {adj} songs, especially energetic ones" if calm > 0.05
                        else f"{adj.capitalize()} songs work for you when they're energetic")
            elif gap < -0.2 and calm > 0.12:
                text = (f"You like {adj} songs, especially calm ones" if energetic > 0.05
                        else f"{adj.capitalize()} songs work for you when they're calm")
            elif min(calm, steady, energetic) > 0.12:
                text = f"You like {adj} songs, calm or energetic"
            elif max(calm, steady, energetic) < -0.12:
                text = f"{adj.capitalize()} songs aren't for you"
            else:
                continue
            found.append((peak + abs(gap) / 2, text))
        found.sort(key=lambda kv: -kv[0])
        texts = [t for _, t in found[:3]]

        themes = sorted(THEME_NAMES, key=lambda t: -self.weight(f"theme:{t}"))
        liked = [t for t in themes[:2] if self.weight(f"theme:{t}") > 0.04]
        disliked = [t for t in themes[::-1][:1] if self.weight(f"theme:{t}") < -0.04]
        if liked:
            texts.append("Lyrics about " + " and ".join(liked) + " pull you in")
        if disliked:
            texts.append(f"Lyrics about {disliked[0]} put you off")
        tone = self.weight("tone")
        if abs(tone) > 0.05:
            texts.append("You lean toward brighter-sounding lyrics" if tone > 0
                         else "You lean toward darker-sounding lyrics")
        return texts[:6]


def feature_phrase(name: str, z: np.ndarray) -> str:
    """Describe what a feature looks like *for this song* (z holds its standardised main effects)."""
    energetic = z[1] > 0
    if name == "tone":
        return "bright tone" if z[0] > 0 else "dark tone"
    if name == "energy":
        return "energetic" if energetic else "calm"
    if name == "tone*energy":
        return ("bright" if z[0] > 0 else "dark") + (" and energetic" if energetic else " and calm")
    if name.startswith("emo:"):
        emotion = name[4:].split("*")[0]
        idx = 2 + EMOTIONS.index(emotion)
        adj = EMOTION_ADJ[emotion]
        if name.endswith("*energy"):
            return (adj if z[idx] > 0 else f"not very {adj}") + (" and energetic" if energetic else " and calm")
        return f"strongly {adj}" if z[idx] > 0 else f"not very {adj}"
    if name.startswith("theme:"):
        theme = name[6:]
        idx = 2 + len(EMOTIONS) + THEME_NAMES.index(theme)
        return f"lyrics about {theme}" if z[idx] > 0 else f"little {theme} in the lyrics"
    return name


# --- Re-ranking --------------------------------------------------------------------------

def _ranks(values: list) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def spearman(a: list, b: list) -> float | None:
    """Spearman rank correlation of two equally long rank lists."""
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def personalize(index, sq, events: list, k: int, deck_size: int = 12) -> dict:
    """Re-rank every song for scored query ``sq`` using the learned taste.

    Returns base and personal top-K track lists, a comparison block, the taste
    profile summary and the next songs to rate ("deck")."""
    n = len(index.songs)
    k = max(1, min(int(k), n))
    base_tracks = {j: index.track(sq, j, 0) for j in range(n)}
    population = [raw_features(t) for t in base_tracks.values()]
    model = PreferenceModel(events, FeatureSpace(population + [s["features"] for s in song_states(events).values()]),
                            neural=True)
    alpha = model.alpha
    states = model.states

    base_rank = {j: r for r, j in enumerate(sq.order, start=1)}
    if model.n_labeled:
        scored = model.preferences(population)
    else:
        zeros = np.zeros(n)
        scored = {"taste": zeros, "linear": zeros, "neural": None}
    personal_score, pref = {}, {}
    for j, track in base_tracks.items():
        pref[j] = float(scored["taste"][j])
        state = states.get(track["key"])
        skips = min(state["skips"], 3) if state else 0
        personal_score[j] = ((1 - alpha) * float(sq.similarity[j]) + alpha * (pref[j] + 1) / 2
                             - SKIP_PENALTY * skips)

    excluded = {j for j, t in base_tracks.items() if states.get(t["key"], {}).get("disliked")}
    personal_order = sorted((j for j in range(n) if j not in excluded),
                            key=lambda j: (-personal_score[j], base_rank[j]))
    personal_rank = {j: r for r, j in enumerate(personal_order, start=1)}

    def decorate(j: int, rank: int) -> dict:
        t = dict(base_tracks[j])
        state = states.get(t["key"]) or {}
        t.update({
            "rank": rank,
            "base_rank": base_rank[j],
            "personal_rank": personal_rank.get(j),
            "rank_change": (base_rank[j] - personal_rank[j]) if j in personal_rank else None,
            "base_percent": int(round(100 * float(sq.similarity[j]))),
            "personal_percent": int(round(100 * max(0.0, min(1.0, personal_score[j])))),
            "preference": round(pref[j], 3),
            "taste_parts": {
                "linear": round(float(scored["linear"][j]), 3),
                "neural": None if scored["neural"] is None else round(float(scored["neural"][j]), 3),
            },
            "reasons": model.contributions(raw_features(t)) if model.n_labeled else [],
            "state": {
                "liked": bool(state.get("liked")), "disliked": bool(state.get("disliked")),
                "saved": bool(state.get("saved")), "skips": int(state.get("skips", 0)),
            },
        })
        return t

    base_top = [decorate(j, r) for r, j in enumerate(sq.order[:k], start=1)]
    personal_top = [decorate(j, r) for r, j in enumerate(personal_order[:k], start=1)]

    base_keys = {t["key"] for t in base_top}
    personal_keys = {t["key"] for t in personal_top}
    dropped = []
    for t in base_top:
        if t["key"] in personal_keys:
            continue
        if t["state"]["disliked"]:
            why = "you marked it not for you"
        elif t["state"]["skips"]:
            why = "you skipped it"
        else:
            why = "other songs fit your taste better"
        dropped.append({"key": t["key"], "title": t["title"], "artist": t["artist"],
                        "base_rank": t["base_rank"], "personal_rank": t["personal_rank"], "why": why})

    common = [j for j in personal_order]
    rho = spearman([base_rank[j] for j in common], [personal_rank[j] for j in common])

    # Next songs to rate: relevant to this mood, not rated yet, and the model is least sure about.
    unrated = [j for j in range(n) if base_tracks[j]["key"] not in states]
    deck_order = sorted(unrated, key=lambda j: -(0.6 * float(sq.similarity[j]) + 0.4 * (1 - abs(pref[j]))))
    deck = [decorate(j, i + 1) for i, j in enumerate(deck_order[:deck_size])]

    return {
        "k": k,
        "alpha": round(alpha, 3),
        "base": base_top,
        "personal": personal_top,
        "deck": deck,
        "comparison": {
            "overlap": round(len(base_keys & personal_keys) / k, 3),
            "spearman": None if rho is None else round(rho, 3),
            "moved": sum(1 for t in personal_top if t["rank_change"]),
            "new_in_top": [t["key"] for t in personal_top if t["key"] not in base_keys],
            "dropped": dropped,
            "excluded": len(excluded),
        },
        "profile": model.summary(),
        "model": model,
        "population": population,
        "ranking": [base_tracks[j]["key"] for j in personal_order],   # every non-hidden song, best first
    }


# --- History ------------------------------------------------------------------------------

def profile_history(events: list, population: list, window: int = HISTORY_WINDOW) -> list:
    """Profile snapshot after each of the last ``window`` events, for the evolution chart."""
    space = FeatureSpace(population + [e.features for e in events])
    start = max(0, len(events) - window)
    history = []
    for i in range(start, len(events)):
        model = PreferenceModel(events[: i + 1], space)
        scale = float(np.max(np.abs(model.w))) or 1.0
        ev = events[i]
        history.append({
            "step": i + 1,
            "action": ev.action,
            "title": ev.title,
            "artist": ev.artist,
            "strength": round(model.alpha / ALPHA_MAX, 3),
            "weights": {n: round(model.weight(n), 4) for n in FEATURES},
            "emotions": {e: round(model.weight(f"emo:{e}") / scale, 3) for e in EMOTIONS},
        })
    return history


def latest_change(history: list, top: int = 3) -> list:
    """The features whose weight moved the most with the latest event."""
    if not history:
        return []
    now = history[-1]["weights"]
    before = history[-2]["weights"] if len(history) > 1 else {n: 0.0 for n in now}
    deltas = sorted(FEATURES, key=lambda n: -abs(now[n] - before[n]))
    return [{"feature": n, "label": feature_label(n), "delta": round(now[n] - before[n], 4)}
            for n in deltas[:top] if abs(now[n] - before[n]) > 1e-4]


def feature_label(name: str) -> str:
    """Neutral, human name for a feature (for change logs and charts)."""
    if name == "tone":
        return "bright tone"
    if name == "energy":
        return "energy"
    if name == "tone*energy":
        return "bright + energetic"
    if name.startswith("emo:"):
        emotion = name[4:].split("*")[0]
        return f"{EMOTION_ADJ[emotion]} + energetic" if name.endswith("*energy") else f"{EMOTION_ADJ[emotion]}"
    if name.startswith("theme:"):
        return f"{name[6:]} lyrics"
    return name


# --- Persistence --------------------------------------------------------------------------

class ProfileStore:
    """Feedback events saved as JSON on the user's machine (no lyrics are stored)."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return [FeedbackEvent.from_dict(d) for d in data.get("events", []) if d.get("action") in ACTIONS]
        except (OSError, ValueError, TypeError):
            return []

    def save(self, events: list) -> None:
        payload = {"version": 1, "events": [asdict(e) for e in events[-MAX_STORED_EVENTS:]]}
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".profile-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass
