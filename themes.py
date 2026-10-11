"""
Lyrical themes: a small, interpretable set of topics a song's lyrics can be "about".

Each theme is a hand-written keyword list. A song's score for a theme is a
saturating function of how often those words appear per 100 lyric tokens, so a
song that mentions rain once scores low and a song that keeps coming back to it
scores high. Scores are in [0, 1] (1.0 only after rounding, for lyrics that barely talk about anything else).

These are deliberately coarse. They exist so the preference model can learn
statements a person can read ("you like lyrics about rain and memories"),
which a 5,000-dimensional TF-IDF vector can't give you from a dozen ratings.
"""

from __future__ import annotations

import math
import re

THEMES = {
    "rain & weather": {"rain", "raining", "rainy", "storm", "storms", "clouds", "cloudy", "thunder", "grey",
                       "gray", "wind", "snow", "umbrella", "raindrops", "drops", "weather", "fog", "lightning"},
    "memories": {"remember", "remembered", "memories", "memory", "yesterday", "ago", "used", "childhood",
                 "photograph", "photographs", "nostalgia", "years", "miss", "once", "back", "old", "young"},
    "love & romance": {"love", "loving", "lover", "kiss", "darling", "baby", "hold", "touch", "forever",
                       "romance", "sweet", "honey", "heart"},
    "heartbreak": {"goodbye", "broken", "tears", "cry", "crying", "leave", "left", "lies", "lie", "hurt",
                   "gone", "over", "without", "ex", "apart"},
    "night & city": {"night", "midnight", "city", "streets", "street", "lights", "neon", "tonight",
                     "downtown", "highway", "car", "drive", "driving"},
    "party & dance": {"dance", "dancing", "party", "club", "floor", "jump", "loud", "bounce", "beat",
                      "drink", "drinks", "celebrate", "music", "move"},
    "rebellion": {"fight", "fighting", "free", "freedom", "rules", "system", "break", "against", "war",
                  "rebel", "revolution", "riot", "chains", "burn"},
    "dreams & sky": {"dream", "dreams", "dreaming", "sky", "stars", "star", "moon", "fly", "flying",
                     "float", "floating", "heaven", "space", "drift", "drifting"},
    "home & comfort": {"home", "warm", "safe", "family", "together", "arms", "bed", "kitchen", "morning",
                       "coffee", "tea", "sunday", "blanket"},
    "loneliness": {"alone", "lonely", "empty", "nobody", "silence", "quiet", "lost", "cold", "nowhere",
                   "lonesome", "isolated"},
    "hope & growth": {"hope", "believe", "rise", "tomorrow", "future", "stronger", "light", "change",
                      "better", "someday", "higher", "grow", "new"},
    "fear & darkness": {"fear", "afraid", "scared", "dark", "darkness", "shadow", "shadows", "ghost",
                        "demon", "demons", "blood", "death", "die", "dying", "nightmare"},
    "summer & sun": {"summer", "sun", "sunshine", "beach", "waves", "ocean", "golden", "sea", "sand",
                     "june", "july", "tan"},
}
THEME_NAMES = tuple(THEMES)

_TOKEN_RE = re.compile(r"[a-z]+")
SATURATION = 2.5   # hits per 100 tokens at which a theme scores tanh(1) ~ 0.76


def theme_scores(lyrics: str) -> dict:
    """Theme name -> score in [0, 1]. Section markers like [Chorus] are ignored."""
    text = re.sub(r"\[[^\]]*\]", " ", (lyrics or "").lower().replace("’", "'"))
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return {name: 0.0 for name in THEME_NAMES}
    scores = {}
    for name, words in THEMES.items():
        hits = sum(1 for tok in tokens if tok in words or (tok.endswith("s") and tok[:-1] in words))
        rate = 100.0 * hits / len(tokens)
        scores[name] = round(math.tanh(rate / SATURATION), 4)
    return scores
