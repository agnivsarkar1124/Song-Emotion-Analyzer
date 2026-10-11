"""
Lyric emotion scoring.

The original approach was ``max(NRCLex(lyrics).raw_emotion_scores, key=...)``.
That has four weaknesses, each addressed below:

1. The raw scores include the lexicon's ``positive`` / ``negative`` polarity
   tags, which are the two most frequent tags by far, so the "top emotion" was
   very often just "positive" or "negative" rather than an actual emotion.
2. Raw counts are biased by how common each emotion is in the lexicon itself
   (e.g. ``fear`` tags nearly three times as many words as ``surprise``), so
   common emotions win by default. Full lift over the base rate overcorrects
   (a couple of rare ``surprise`` hits would beat a dozen ``anger`` hits), so
   we divide by the square root of the base rate instead, after Dirichlet
   smoothing toward that base rate so short lyrics don't produce wild ratios.
3. Negation ("never happy", "don't trust") was counted as the emotion itself.
   Negated hits are now skipped.
4. Hooks repeat. A chorus sung four times shouldn't outvote the rest of the
   song, so repeated lines are down-weighted (1 / sqrt(n) on the n-th repeat),
   and lines VADER finds strongly polarised count a little more than flat ones.

Finally, scores are re-weighted by how well each emotion agrees with the
song's overall VADER tone (a "joy" call on a clearly negative song is
suspicious). That prior is a heuristic and has not been validated against
hand-labelled data yet. Anger vs. fear vs. sadness remain hard to separate
with a word lexicon alone; ``confidence`` / ``ambiguous`` say when the top two
are close. (An arousal prior based on VADER variance was tried and dropped: a
song that is consistently furious has *low* line-to-line variance.)

Pure Python + the NRC lexicon that ``nrclex`` already bundles; no model
downloads and no TextBlob/NLTK corpora needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
_EMOTION_SET = frozenset(EMOTIONS)

# Rough valence of each emotion in [-1, 1], used for the tone-consistency prior.
EMOTION_VALENCE = {
    "joy": 0.9, "trust": 0.6, "anticipation": 0.3, "surprise": 0.0,
    "anger": -0.7, "fear": -0.7, "sadness": -0.9, "disgust": -0.6,
}

NEGATORS = frozenset({
    "not", "no", "never", "nothing", "nobody", "neither", "nor", "without", "cannot",
    "dont", "don't", "cant", "can't", "wont", "won't", "isnt", "isn't", "aint", "ain't",
    "didnt", "didn't", "wasnt", "wasn't", "arent", "aren't", "couldnt", "couldn't",
    "wouldnt", "wouldn't", "shouldnt", "shouldn't", "doesnt", "doesn't",
})

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SECTION_RE = re.compile(r"^\[.*\]$")  # Genius section markers: [Chorus], [Verse 1]

_vader = None


def _vader_compound(line: str) -> float:
    """Line-level VADER compound score, or 0.0 if vaderSentiment isn't installed."""
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            _vader = False
    if _vader is False:
        return 0.0
    return _vader.polarity_scores(line)["compound"]


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    """word -> tuple of emotions (polarity tags dropped). Empty if nrclex is missing."""
    try:
        from nrclex import NRCLex
    except ImportError:
        return {}
    lexicon = {}
    for word, tags in NRCLex.lexicon.items():
        emotions = tuple(t for t in tags if t in _EMOTION_SET)
        if emotions:
            lexicon[word] = emotions
    return lexicon


def lexicon_available() -> bool:
    return bool(load_lexicon())


@lru_cache(maxsize=1)
def emotion_priors() -> dict:
    """Base rate of each emotion across the lexicon's (word, emotion) pairs."""
    counts = Counter(e for emotions in load_lexicon().values() for e in emotions)
    total = sum(counts.values())
    if not total:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    return {e: max(counts[e], 1) / total for e in EMOTIONS}


def _candidates(tok: str):
    """Surface form first, then crude inflection stripping (loved -> love, tears -> tear)."""
    yield tok
    if tok.endswith("in"):
        yield tok + "g"                      # lovin -> loving
    if tok.endswith("'s"):
        yield tok[:-2]
    if tok.endswith("ing") and len(tok) > 5:
        yield tok[:-3]
        yield tok[:-3] + "e"
    if tok.endswith("ed") and len(tok) > 4:
        yield tok[:-2]
        yield tok[:-1]
    if tok.endswith("ies") and len(tok) > 4:
        yield tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 4:
        yield tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        yield tok[:-1]


def _lookup(tok: str, lexicon: dict):
    for cand in _candidates(tok):
        if len(cand) >= 3 and cand in lexicon:
            return lexicon[cand]
    return None


def _empty_result(coverage: float = 0.0) -> dict:
    return {
        "top_emotion": None,
        "secondary_emotion": None,
        "confidence": 0.0,
        "ambiguous": True,
        "profile": {e: 0.0 for e in EMOTIONS},
        "raw_counts": {e: 0.0 for e in EMOTIONS},
        "coverage": round(coverage, 4),
        "negated_hits": 0,
    }


def analyze_lyrics(
    text: str,
    valence: float | None = None,
    *,
    kappa: float = 5.0,
    beta: float = 0.5,
    gamma: float = 0.6,
    repeat_decay: float = 0.5,
) -> dict:
    """
    Score the eight NRC emotions for ``text``.

    ``valence`` is the song's standardised VADER tone (negative = dark,
    positive = bright, roughly in [-1.5, 1.5]); pass it to enable the
    tone-consistency prior. ``kappa`` is the smoothing strength toward the
    lexicon base rate, ``beta`` how much of the base-rate correction to apply
    (0 = raw shares, 1 = full lift), ``gamma`` the strength of the tone prior,
    ``repeat_decay`` the exponent on repeated-line damping.

    Returns a dict with ``top_emotion`` / ``secondary_emotion`` (None when no
    lexicon words matched), ``confidence`` (relative margin between the top two,
    0..1), ``profile`` (8 shares summing to 1), ``raw_counts`` (weighted hits),
    ``coverage`` (fraction of tokens that hit the lexicon) and ``negated_hits``.
    """
    lexicon = load_lexicon()
    if not lexicon or not text:
        return _empty_result()

    counts = Counter()
    seen_lines = Counter()
    total_tokens = hit_tokens = negated_hits = 0

    for raw_line in text.replace("’", "'").split("\n"):
        line = raw_line.strip().lower()
        if not line or _SECTION_RE.match(line):
            continue
        tokens = _TOKEN_RE.findall(line)
        if not tokens:
            continue
        seen_lines[line] += 1
        repeat_weight = seen_lines[line] ** -repeat_decay
        weight = repeat_weight * (1.0 + abs(_vader_compound(line)))

        for i, tok in enumerate(tokens):
            total_tokens += 1
            emotions = _lookup(tok, lexicon)
            if not emotions:
                continue
            if any(prev in NEGATORS for prev in tokens[max(0, i - 2):i]):
                negated_hits += 1
                continue
            hit_tokens += 1
            for emotion in emotions:
                counts[emotion] += weight

    coverage = hit_tokens / total_tokens if total_tokens else 0.0
    mass = sum(counts.values())
    if mass <= 0:
        result = _empty_result(coverage)
        result["negated_hits"] = negated_hits
        return result

    priors = emotion_priors()
    scores = {}
    for e in EMOTIONS:
        smoothed_share = (counts[e] + kappa * priors[e]) / (mass + kappa)
        score = smoothed_share / (priors[e] ** beta)
        if valence is not None:
            score *= math.exp(gamma * EMOTION_VALENCE[e] * max(-1.5, min(1.5, valence)))
        scores[e] = score

    norm = sum(scores.values())
    profile = {e: scores[e] / norm for e in EMOTIONS}
    ranked = sorted(EMOTIONS, key=lambda e: profile[e], reverse=True)
    top, second = ranked[0], ranked[1]
    confidence = (profile[top] - profile[second]) / profile[top] if profile[top] else 0.0

    return {
        "top_emotion": top,
        "secondary_emotion": second,
        "confidence": round(confidence, 4),
        "ambiguous": confidence < 0.08,
        "profile": {e: round(profile[e], 4) for e in EMOTIONS},
        "raw_counts": {e: round(counts[e], 3) for e in EMOTIONS},
        "coverage": round(coverage, 4),
        "negated_hits": negated_hits,
    }


def legacy_top_emotion(text: str) -> str | None:
    """The original behaviour (argmax of raw lexicon counts, polarity tags included).
    Kept only so tests and docs can compare old vs new."""
    try:
        from nrclex import NRCLex
    except ImportError:
        return None
    counts = Counter()
    for tok in _TOKEN_RE.findall(text.lower().replace("’", "'")):
        for tag in NRCLex.lexicon.get(tok, ()):
            counts[tag] += 1
    return max(counts, key=counts.get) if counts else None
