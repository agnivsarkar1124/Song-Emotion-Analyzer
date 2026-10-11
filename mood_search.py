"""
Free-text mood search over the song library (retrieval half of the RAG pipeline).

    "chill but slightly nostalgic for a rainy day"
        -> MoodQuery  (a point in mood space + weighted lyric keywords)
        -> MoodIndex.search()  (distance to every song, top-K)
        -> JSON playlist

Every song and every query is mapped into the same interpretable *mood space*:

    valence    standardised VADER mean   (dark  <-> bright)
    intensity  standardised VADER variance (steady <-> volatile)
    emotions   8-way NRC emotion profile (anger ... trust), sums to 1
    themes     TF-IDF vector over the lyrics (query side: expanded keywords)

Distance to a song is ``1 - S`` where S is a weighted blend of three similarities,
each in [0, 1]:

    S_emotion  cosine(query emotion profile, song emotion profile)
    S_mood     exp(-d^2 / 2*sigma^2),  d = Euclidean distance in (valence, intensity)
    S_theme    tanh(cosine(query keywords, song TF-IDF) / 0.12)

This is deliberately *not* a neural embedding: sentence-transformers has no
PyTorch wheel on every platform (see the README), and an interpretable space lets
us explain *why* a track matched. A mood lexicon ("rainy" -> sadness, calm,
keywords rain/storm/grey ...) plays the role of query expansion.

Nothing here imports Streamlit, so it is unit-testable on its own.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from emotion_engine import EMOTIONS, analyze_lyrics, load_lexicon
from themes import theme_scores

SCHEMA_VERSION = 1
DEFAULT_WEIGHTS = {"emotion": 0.35, "mood": 0.40, "theme": 0.25}
MOOD_SIGMA = 1.0           # width of the (valence, intensity) kernel, in standardised units
THEME_SATURATION = 0.12    # cosine at which the theme similarity reaches tanh(1) ~ 0.76
MAX_QUERY_CHARS = 2000


# --- Calibration ---------------------------------------------------------------------

def _clip(x: float, lo: float = -1.5, hi: float = 1.5) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Calibration:
    """Maps raw (mood, variance) from get_song_mood() to standardised axes.

    Derived from the app's percentile-calibrated CATEGORY_TARGETS so that the
    five category targets land roughly at +/-1 on each axis and the typical song
    near the origin.
    """
    mood_center: float
    mood_scale: float
    var_center: float
    var_scale: float

    @classmethod
    def from_targets(cls, targets: dict) -> "Calibration":
        moods = [t["mood"] for t in targets.values()]
        variances = [t["variance"] for t in targets.values()]
        return cls(
            mood_center=statistics.median(moods),
            mood_scale=max((max(moods) - min(moods)) / 2, 1e-6),
            var_center=statistics.median(variances),
            var_scale=max((max(variances) - min(variances)) / 2, 1e-6),
        )

    def valence(self, mood: float) -> float:
        return _clip((mood - self.mood_center) / self.mood_scale)

    def intensity(self, variance: float) -> float:
        return _clip((variance - self.var_center) / self.var_scale)


# --- Mood lexicon (query understanding) ----------------------------------------------
#
# Each entry: valence / intensity in [-1, 1] on the same axes as above, an emotion
# mix (any scale; normalised later) and lyric keywords used for query expansion.
# Hand-written, so extend freely: a new row is all it takes to teach it a new mood.

_CONCEPTS = [
    (("chill", "relaxed", "calm", "mellow", "laid-back", "laidback", "peaceful", "easygoing",
      "soothing", "gentle", "lazy", "sleepy", "serene", "unwind", "relaxing", "smooth"),
     dict(v=0.3, i=-1.0, emo={"trust": .5, "joy": .3, "anticipation": .1},
          kw=["easy", "slow", "gentle", "rest", "sleep", "lazy", "breeze", "quiet", "soft", "calm",
              "sunday", "morning", "home", "warm", "ease"])),
    (("happy", "joyful", "cheerful", "upbeat", "bright", "fun", "euphoric", "celebratory",
      "uplifting", "feelgood", "feel-good", "positive", "joy", "elated"),
     dict(v=1.0, i=0.0, emo={"joy": .6, "trust": .2, "anticipation": .2},
          kw=["happy", "sunshine", "smile", "dance", "laugh", "alive", "good", "love", "bright",
              "shine", "joy", "sing"])),
    (("sad", "melancholy", "melancholic", "blue", "heartbroken", "gloomy", "somber", "sombre",
      "sorrowful", "depressing", "tearful", "mournful", "grief", "sorrow", "down"),
     dict(v=-1.0, i=-0.2, emo={"sadness": .7, "fear": .1, "disgust": .05},
          kw=["cry", "tears", "alone", "broken", "goodbye", "hurt", "lonely", "pain", "heart",
              "gone", "lost", "sorrow"])),
    (("nostalgic", "nostalgia", "wistful", "reminiscent", "sentimental", "bittersweet",
      "homesick", "reminiscing", "throwback"),
     dict(v=-0.1, i=-0.5, emo={"sadness": .35, "trust": .25, "anticipation": .15, "joy": .25},
          kw=["remember", "memories", "memory", "yesterday", "old", "young", "used", "back", "time",
              "gone", "miss", "days", "summer", "childhood", "once", "years", "ago", "home",
              "photograph", "photographs", "remembered"])),
    (("rainy", "rain", "stormy", "storm", "grey", "gray", "overcast", "drizzle", "cloudy", "rainstorm"),
     dict(v=-0.3, i=-0.5, emo={"sadness": .4, "fear": .1, "trust": .1, "anticipation": .1},
          kw=["rain", "raindrops", "storm", "clouds", "grey", "gray", "window", "thunder", "wet",
              "drops", "umbrella", "cold", "puddles"])),
    (("angry", "furious", "rage", "aggressive", "pissed", "hostile", "bitter", "resentful",
      "mad", "wrathful", "livid", "vengeful"),
     dict(v=-0.8, i=1.0, emo={"anger": .7, "disgust": .2, "fear": .1},
          kw=["hate", "rage", "fight", "burn", "scream", "kill", "break", "war", "blood", "mad",
              "hell", "destroy"])),
    (("intense", "powerful", "epic", "explosive", "dramatic", "anthemic", "cathartic", "fierce"),
     dict(v=0.2, i=1.0, emo={"anticipation": .3, "anger": .2, "joy": .2, "fear": .1, "surprise": .1},
          kw=["fire", "rise", "power", "strong", "fight", "believe", "higher", "break", "lightning",
              "thunder"])),
    (("romantic", "loving", "tender", "passionate", "intimate", "sweet", "lovey", "affectionate"),
     dict(v=0.6, i=-0.2, emo={"joy": .3, "trust": .5, "anticipation": .2},
          kw=["love", "kiss", "heart", "hold", "darling", "baby", "forever", "touch", "eyes", "night",
              "tonight"])),
    (("lonely", "isolated", "solitary", "empty", "hollow", "alone"),
     dict(v=-0.8, i=-0.6, emo={"sadness": .6, "fear": .2, "disgust": .1},
          kw=["alone", "empty", "lonely", "nobody", "silence", "night", "dark", "cold", "apart"])),
    (("hopeful", "optimistic", "inspiring", "motivated", "determined", "empowering", "motivational",
      "inspirational", "triumphant"),
     dict(v=0.6, i=0.3, emo={"anticipation": .4, "joy": .3, "trust": .3},
          kw=["tomorrow", "hope", "rise", "believe", "dream", "future", "better", "light", "new",
              "someday", "stronger"])),
    (("anxious", "nervous", "tense", "paranoid", "uneasy", "restless", "worried", "afraid",
      "scared", "fearful"),
     dict(v=-0.6, i=0.5, emo={"fear": .6, "anticipation": .2, "surprise": .1},
          kw=["afraid", "scared", "fear", "shaking", "nervous", "run", "dark", "anxious", "breath",
              "shadow", "shadows"])),
    (("dreamy", "ethereal", "floaty", "hazy", "atmospheric", "spacey", "dreaming", "otherworldly"),
     dict(v=0.1, i=-0.8, emo={"anticipation": .3, "joy": .2, "trust": .3, "surprise": .2},
          kw=["dream", "sky", "float", "stars", "clouds", "sleep", "fade", "haze", "moon", "drift",
              "floating", "drifting"])),
    (("moody", "brooding", "dark", "late-night", "late night", "midnight", "nighttime", "nocturnal"),
     dict(v=-0.4, i=-0.2, emo={"fear": .2, "sadness": .3, "anticipation": .2, "anger": .1},
          kw=["night", "midnight", "dark", "moon", "shadows", "streets", "lights", "hours", "city"])),
    (("summery", "summer", "beach", "carefree", "sunny", "sunshine", "road trip", "roadtrip", "tropical"),
     dict(v=0.7, i=-0.3, emo={"joy": .5, "anticipation": .2, "trust": .2},
          kw=["summer", "sun", "beach", "waves", "warm", "sunshine", "drive", "windows", "golden",
              "road"])),
    (("party", "dance", "danceable", "energetic", "hype", "hyped", "pumped", "workout", "bouncy",
      "club", "banger"),
     dict(v=0.7, i=0.8, emo={"joy": .5, "anticipation": .3, "surprise": .1},
          kw=["dance", "party", "night", "move", "jump", "loud", "tonight", "bounce", "floor", "hands",
              "body"])),
    (("focus", "study", "studying", "ambient", "lofi", "lo-fi", "lo fi", "background", "concentration"),
     dict(v=0.1, i=-1.0, emo={"trust": .4, "anticipation": .2, "joy": .2},
          kw=["easy", "slow", "quiet", "soft", "calm", "flow"])),
    (("heartbreak", "breakup", "break-up", "betrayal", "ex", "dumped", "heartache"),
     dict(v=-0.9, i=0.1, emo={"sadness": .5, "anger": .2, "disgust": .1, "fear": .1},
          kw=["goodbye", "left", "leave", "broken", "lies", "cry", "over", "hurt", "tears", "without",
              "heart"])),
    (("cozy", "cosy", "warm", "comforting", "wholesome", "comfy", "homey"),
     dict(v=0.6, i=-0.7, emo={"trust": .5, "joy": .3, "anticipation": .1},
          kw=["home", "warm", "fire", "tea", "blanket", "hold", "safe", "together", "arms"])),
    (("rebellious", "defiant", "rebel", "punk", "revolt", "protest"),
     dict(v=-0.2, i=0.9, emo={"anger": .4, "anticipation": .2, "joy": .1},
          kw=["fight", "system", "rise", "free", "against", "break", "rules", "run", "freedom"])),
    (("spooky", "eerie", "creepy", "haunting", "haunted", "ominous", "sinister"),
     dict(v=-0.6, i=0.2, emo={"fear": .6, "disgust": .1, "surprise": .1},
          kw=["dark", "ghost", "shadow", "cold", "scream", "fear", "blood", "demon", "haunted"])),
]

_CONCEPT_INDEX: dict = {}
for _terms, _spec in _CONCEPTS:
    for _term in _terms:
        _CONCEPT_INDEX.setdefault(_term, _spec)

_MODIFIERS = {
    # diminishers (< 1) pull a concept toward the neutral centre; intensifiers (> 1) push outward
    "slightly": .45, "somewhat": .6, "kinda": .6, "mildly": .5, "faintly": .4, "barely": .3,
    "bit": .5, "little": .5, "touch": .4, "hint": .4, "tad": .4, "lightly": .5, "vaguely": .4,
    "kind of": .6, "sort of": .6, "a bit": .5, "a little": .5, "a touch": .4, "a hint": .4,
    "very": 1.5, "really": 1.4, "extremely": 1.8, "super": 1.5, "incredibly": 1.7, "deeply": 1.6,
    "totally": 1.5, "so": 1.3, "quite": 1.2, "intensely": 1.7, "utterly": 1.7, "especially": 1.4,
}
_NEGATORS = frozenset({"not", "no", "never", "without", "non", "isn't", "dont", "don't", "nothing", "less"})
_STOPWORDS = frozenset(
    "a an and the of for to in on at it is its be but or with that this these those some any "
    "i me my we our you your want something feel feeling feels like day days mood vibe vibes "
    "songs song music playlist kind sort bit little very really so quite just more most "
    "slightly somewhat mildly barely touch hint tad lightly vaguely extremely super incredibly "
    "deeply totally intensely utterly especially kinda".split()
)
_QUERY_TOKEN_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


# --- Query parsing -----------------------------------------------------------------

@dataclass
class MoodQuery:
    text: str
    valence: float
    intensity: float
    emotions: dict           # 8 shares summing to 1 (all zero if no emotion evidence)
    keywords: dict           # lyric keyword -> weight
    concepts: list           # [{"term", "weight", "negated"}]
    used_fallback: bool      # no lexicon concepts matched; sentiment/NRC only
    notes: list

    @property
    def has_affect(self) -> bool:
        return bool(self.concepts) or any(self.emotions.values()) or abs(self.valence) > 0.05


def parse_mood_description(text: str) -> MoodQuery:
    """Turn a free-text mood description into a point in mood space plus keywords."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Describe the mood you want first, e.g. 'chill but slightly nostalgic for a rainy day'.")
    text = text[:MAX_QUERY_CHARS]
    tokens = _QUERY_TOKEN_RE.findall(text.lower().replace("’", "'"))

    hits = []                    # (spec, term, weight, negated)
    pending, gap = 1.0, 0
    i = 0
    while i < len(tokens):
        bigram = " ".join(tokens[i:i + 2]) if i + 1 < len(tokens) else None
        tok = tokens[i]
        step = 1

        if bigram in _MODIFIERS:
            factor, step = _MODIFIERS[bigram], 2
            pending = min(pending, factor) if factor < 1 else max(pending, factor)
            gap = 0
        elif tok in _MODIFIERS:
            factor = _MODIFIERS[tok]
            pending = min(pending, factor) if factor < 1 else max(pending, factor)
            gap = 0
        else:
            spec, term = None, None
            if bigram in _CONCEPT_INDEX:
                spec, term, step = _CONCEPT_INDEX[bigram], bigram, 2
            elif tok in _CONCEPT_INDEX:
                spec, term = _CONCEPT_INDEX[tok], tok
            elif tok.endswith("s") and tok[:-1] in _CONCEPT_INDEX:
                spec, term = _CONCEPT_INDEX[tok[:-1]], tok[:-1]
            if spec is not None:
                negated = any(t in _NEGATORS for t in tokens[max(0, i - 2):i])
                hits.append((spec, term, pending, negated))
                pending, gap = 1.0, 0
            else:
                gap += 1
                if gap > 2:
                    pending = 1.0
        i += step

    notes = []
    used_fallback = not hits
    live = [(s, t, w) for s, t, w, neg in hits if not neg]
    flipped = [(s, t, w) for s, t, w, neg in hits if neg]

    emotions = {e: 0.0 for e in EMOTIONS}
    keywords: dict = {}

    if hits:
        total_w = sum(w for _, _, w in live) + sum(w for _, _, w in flipped)
        # Direction = weighted mean of concept positions; magnitude follows the strongest
        # modifier, so "slightly nostalgic" sits closer to neutral than "very nostalgic".
        v_sum = sum(w * s["v"] for s, _, w in live) - 0.6 * sum(w * s["v"] for s, _, w in flipped)
        i_sum = sum(w * s["i"] for s, _, w in live) - 0.6 * sum(w * s["i"] for s, _, w in flipped)
        magnitude = _clip(max(w for _, _, w, _ in hits), 0.3, 1.6)
        valence = _clip(v_sum / total_w * magnitude)
        intensity = _clip(i_sum / total_w * magnitude)
        for spec, _, w in live:
            for e, share in spec["emo"].items():
                emotions[e] += w * share
            for kw in spec["kw"]:
                keywords[kw] = max(keywords.get(kw, 0.0), w)
        if flipped:
            notes.append("Negated terms ("
                         + ", ".join(t for _, t, _ in flipped)
                         + ") push the mood the opposite way.")
    else:
        valence = intensity = 0.0
        lexicon = load_lexicon()
        for tok in tokens:
            for e in lexicon.get(tok, ()):
                emotions[e] += 0.5
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            valence = _clip(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
        except ImportError:
            pass
        notes.append("No known mood words found; falling back to the sentiment/emotion lexicons "
                     "and plain keyword overlap. Try words like calm, nostalgic, rainy, angry.")

    mass = sum(emotions.values())
    if mass > 0:
        emotions = {e: v / mass for e, v in emotions.items()}

    # Raw words the user typed count as keywords too (they may literally appear in lyrics).
    for tok in tokens:
        if len(tok) >= 3 and tok not in _STOPWORDS and tok not in _MODIFIERS and tok not in _NEGATORS:
            keywords[tok] = max(keywords.get(tok, 0.0), 0.7)

    concepts = [{"term": t, "weight": round(w, 3), "negated": neg} for _, t, w, neg in hits]
    return MoodQuery(
        text=text,
        valence=round(valence, 4),
        intensity=round(intensity, 4),
        emotions={e: round(v, 4) for e, v in emotions.items()},
        keywords=keywords,
        concepts=concepts,
        used_fallback=used_fallback,
        notes=notes,
    )


# --- Index + search ------------------------------------------------------------------

def song_key(title: str, artist: str) -> str:
    """Stable identifier for a song across sessions (case-insensitive title|artist)."""
    return f"{title.strip().lower()}|{artist.strip().lower()}"


@dataclass
class ScoredQuery:
    query: MoodQuery
    weights: dict             # active channel weights, sum to 1
    similarity: np.ndarray    # combined similarity per song, in [0, 1]
    s_emotion: np.ndarray
    s_mood: np.ndarray
    s_theme: np.ndarray
    used_kw: dict             # query keywords found in the lyric vocabulary -> column
    order: list               # song indices, most similar first


def _describe_axes(valence: float, intensity: float) -> str:
    tone = "bright" if valence > 0.35 else "dark" if valence < -0.35 else "bittersweet" if valence < -0.1 else "neutral-toned"
    energy = "volatile" if intensity > 0.5 else "calm" if intensity < -0.5 else "steady"
    return f"{energy}, {tone}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _safe_url(url):
    return url if isinstance(url, str) and url.startswith("https://") else None


class MoodIndex:
    """In-memory index over song dicts shaped like ``add_song_to_library()`` output
    (needs title, artist, lyrics, mood, variance; everything else is optional)."""

    def __init__(self, songs: list, calibration: Calibration):
        self.calibration = calibration
        self.songs = []
        for song in songs:
            valence = calibration.valence(song["mood"])
            intensity = calibration.intensity(song["variance"])
            analysis = song.get("emotion_analysis") or analyze_lyrics(song["lyrics"], valence)
            profile = np.array([analysis["profile"].get(e, 0.0) for e in EMOTIONS], dtype=float)
            self.songs.append({
                "src": song, "key": song_key(song["title"], song["artist"]),
                "valence": valence, "intensity": intensity,
                "analysis": analysis, "profile": profile,
                "themes": song.get("themes") or theme_scores(song["lyrics"]),
            })

        self.vectorizer = None
        self.matrix = None
        docs = [s["src"]["lyrics"] for s in self.songs]
        if docs:
            try:
                self.vectorizer = TfidfVectorizer(
                    stop_words="english", sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
                )
                self.matrix = self.vectorizer.fit_transform(docs)
            except ValueError:   # empty vocabulary (e.g. all lyrics are stop words)
                self.vectorizer = self.matrix = None

    def __len__(self):
        return len(self.songs)

    def _query_vector(self, query: MoodQuery):
        """Sparse-aligned dense TF-IDF vector for the expanded keywords, plus its vocabulary hits."""
        if self.vectorizer is None:
            return None, {}
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_
        vec = np.zeros(len(vocab))
        used = {}
        for kw, weight in query.keywords.items():
            idx = vocab.get(kw)
            if idx is not None:
                vec[idx] += weight * idf[idx]
                used[kw] = idx
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else None), used

    def score_all(self, description: str, weights: dict | None = None) -> "ScoredQuery":
        """Similarity of *every* song to the description. ``search`` takes the top K of this;
        personalisation re-ranks the full list."""
        query = parse_mood_description(description)
        n = len(self.songs)
        if n == 0:
            raise ValueError("The song database is empty - add songs or enable the seed songs first.")

        base = dict(DEFAULT_WEIGHTS)
        if weights:
            base.update(weights)

        q_emo = np.array([query.emotions[e] for e in EMOTIONS])
        s_emotion = np.array([_cosine(q_emo, s["profile"]) for s in self.songs])

        d2 = np.array([(s["valence"] - query.valence) ** 2 + (s["intensity"] - query.intensity) ** 2
                       for s in self.songs])
        s_mood = np.exp(-d2 / (2 * MOOD_SIGMA ** 2))

        q_vec, used_kw = self._query_vector(query)
        if q_vec is not None:
            raw = np.asarray(self.matrix @ q_vec).ravel()
            s_theme = np.tanh(raw / THEME_SATURATION)
        else:
            s_theme = np.zeros(n)

        active = {
            "emotion": base["emotion"] if q_emo.any() else 0.0,
            "mood": base["mood"] if query.has_affect else 0.0,
            "theme": base["theme"] if q_vec is not None else 0.0,
        }
        total = sum(active.values())
        if total <= 0:
            raise ValueError("Couldn't extract a mood from that description. Try words like "
                             "'calm', 'angry', 'nostalgic', 'rainy' or 'upbeat'.")
        active = {name: w / total for name, w in active.items()}

        similarity = (active["emotion"] * s_emotion + active["mood"] * s_mood + active["theme"] * s_theme)
        order = sorted(range(n), key=lambda j: (-similarity[j], self.songs[j]["src"]["title"].lower()))
        return ScoredQuery(query=query, weights=active, similarity=similarity, s_emotion=s_emotion,
                           s_mood=s_mood, s_theme=s_theme, used_kw=used_kw, order=order)

    def track(self, sq: "ScoredQuery", j: int, rank: int) -> dict:
        """JSON-ready description of song ``j`` for a scored query (no lyrics)."""
        song = self.songs[j]
        src = song["src"]
        reason = self._explain(sq.query, song, j, sq.used_kw, sq.s_emotion[j])
        top_emotion = song["analysis"].get("top_emotion") or src.get("top_emotion")
        similarity = float(sq.similarity[j])
        return {
            "rank": rank,
            "key": song["key"],
            "title": src["title"],
            "artist": src["artist"],
            "distance": round(1.0 - similarity, 4),
            "match_percent": int(round(100 * similarity)),
            "scores": {
                "emotion": round(float(sq.s_emotion[j]), 4),
                "mood": round(float(sq.s_mood[j]), 4),
                "theme": round(float(sq.s_theme[j]), 4),
            },
            "category": src.get("category"),
            "top_emotion": top_emotion if top_emotion not in (None, "-") else None,
            "secondary_emotion": song["analysis"].get("secondary_emotion"),
            "emotion_confidence": song["analysis"].get("confidence"),
            "emotion_profile": {e: round(float(song["profile"][x]), 4) for x, e in enumerate(EMOTIONS)},
            "themes": song["themes"],
            "valence": round(float(song["valence"]), 3),
            "intensity": round(float(song["intensity"]), 3),
            "mood": src["mood"],
            "variance": src["variance"],
            "duration_seconds": src.get("duration_seconds"),
            "preview_url": _safe_url(src.get("preview_url")),
            "track_url": _safe_url(src.get("track_url")),
            "matched_keywords": reason["keywords"],
            "why": reason["text"],
        }

    def rag_hit(self, sq: "ScoredQuery", j: int, score: float | None = None) -> dict:
        """Retrieved song with lyrics, for the grounded-explanation step."""
        src = self.songs[j]["src"]
        return {
            "title": src["title"], "artist": src["artist"], "lyrics": src["lyrics"],
            "mood": src["mood"], "variance": src["variance"],
            "match_score": round(float(sq.similarity[j] if score is None else score), 3),
        }

    def playlist(self, sq: "ScoredQuery", tracks: list) -> dict:
        """Wrap a ranked track list in the playlist JSON envelope."""
        profiles = [[t["emotion_profile"][e] for e in EMOTIONS] for t in tracks] or [[0.0] * len(EMOTIONS)]
        mean_profile = np.mean(profiles, axis=0)
        query = sq.query
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query": query.text,
            "k": len(tracks),
            "database_size": len(self.songs),
            "weights": {name: round(w, 4) for name, w in sq.weights.items()},
            "distance": "1 - (w_emotion*cos(emotions) + w_mood*exp(-d^2/2s^2) + w_theme*tanh(cos(tfidf)/0.12))",
            "parsed_mood": {
                "valence": query.valence,
                "intensity": query.intensity,
                "description": _describe_axes(query.valence, query.intensity),
                "emotions": query.emotions,
                "concepts": query.concepts,
                "keywords": sorted(sq.used_kw, key=lambda w: -query.keywords[w])[:12],
                "used_fallback": query.used_fallback,
                "notes": query.notes,
            },
            "summary": {
                "mean_distance": round(float(np.mean([t["distance"] for t in tracks])), 4) if tracks else None,
                "mean_valence": round(float(np.mean([t["valence"] for t in tracks])), 3) if tracks else 0.0,
                "mean_intensity": round(float(np.mean([t["intensity"] for t in tracks])), 3) if tracks else 0.0,
                "emotion_profile": {e: round(float(mean_profile[x]), 4) for x, e in enumerate(EMOTIONS)},
                "dominant_emotion": EMOTIONS[int(np.argmax(mean_profile))],
            },
            "tracks": tracks,
        }

    def search(self, description: str, k: int = 5, weights: dict | None = None):
        """Return ``(playlist, hits)``: the JSON-serialisable top-K playlist and, for the RAG
        explanation step, the retrieved songs including their lyrics."""
        sq = self.score_all(description, weights)
        k = max(1, min(int(k), len(self.songs)))
        top = sq.order[:k]
        tracks = [self.track(sq, j, rank) for rank, j in enumerate(top, start=1)]
        return self.playlist(sq, tracks), [self.rag_hit(sq, j) for j in top]

    def _explain(self, query: MoodQuery, song: dict, row: int, used_kw: dict, emo_sim: float) -> dict:
        overlap = {
            e: min(query.emotions[e], float(song["profile"][x]))
            for x, e in enumerate(EMOTIONS)
        }
        shared = [e for e, v in sorted(overlap.items(), key=lambda kv: -kv[1]) if v > 0.03][:2]

        keywords = []
        if self.matrix is not None and used_kw:
            row_vec = self.matrix.getrow(row)
            cols = dict(zip(row_vec.indices, row_vec.data))
            scored = [(query.keywords[kw] * cols[idx], kw) for kw, idx in used_kw.items() if idx in cols]
            keywords = [kw for _, kw in sorted(scored, reverse=True)[:5]]

        parts = [f"feels {_describe_axes(song['valence'], song['intensity'])}"]
        if shared:
            parts.append("shares " + " & ".join(shared) + " with your description")
        if keywords:
            parts.append("lyrics touch on " + ", ".join(keywords))
        return {"text": "; ".join(parts).capitalize() + ".", "keywords": keywords}


def playlist_to_json(playlist: dict, indent: int = 2) -> str:
    return json.dumps(playlist, indent=indent, ensure_ascii=False)
