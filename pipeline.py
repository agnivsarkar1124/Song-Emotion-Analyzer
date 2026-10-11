"""
Lyric pipeline shared by the Streamlit app and command-line tools (``benchmark.py``):
fetch lyrics from Genius, score sentiment / emotion, look up track info on iTunes,
and package everything as a song record. Nothing here imports Streamlit.
"""

from __future__ import annotations

import math
import statistics
import time

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from emotion_engine import analyze_lyrics, lexicon_available
from mood_search import Calibration

NRCLEX_AVAILABLE = lexicon_available()

CATEGORY_TARGETS = {
    "Happy":   {"mood": 0.33, "variance": 0.15},
    "Sad":     {"mood": -0.07, "variance": 0.15},
    "Chill":   {"mood": 0.17, "variance": 0.10},
    "Intense": {"mood": 0.17, "variance": 0.23},
    "Angry":   {"mood": -0.07, "variance": 0.23},
}

# Maps raw (mood, variance) onto the standardised valence/intensity axes used by
# the free-text mood search; derived from the calibrated targets above.
CALIBRATION = Calibration.from_targets(CATEGORY_TARGETS)

SEED_SONGS = {
    "Happy": [("Happy", "Pharrell Williams"), ("Good as Hell", "Lizzo"), ("Walking on Sunshine", "Katrina and the Waves")],
    "Sad": [("Someone Like You", "Adele"), ("Skinny Love", "Bon Iver"), ("Someone You Loved", "Lewis Capaldi")],
    "Chill": [("Sunday Morning", "Maroon 5"), ("Banana Pancakes", "Jack Johnson"), ("Redbone", "Childish Gambino")],
    "Intense": [("Believer", "Imagine Dragons"), ("Radioactive", "Imagine Dragons"), ("Numb", "Linkin Park")],
    "Angry": [("Killing in the Name", "Rage Against the Machine"), ("Break Stuff", "Limp Bizkit"), ("Bodies", "Drowning Pool")],
}


def fetch_with_retry(url, headers=None, params=None, retries=3, delay=2):
    last_error = None
    for attempt in range(retries):
        try:
            return requests.get(url, headers=headers, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_error


def get_lyrics(song_title, artist, token, on_note=None):
    """Lyrics for a song from Genius. ``on_note(message)`` is called when the best match's
    artist doesn't match (the app shows it as a caption; scripts can print it)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    }
    params = {"q": f"{song_title} {artist}"}
    response = fetch_with_retry("https://api.genius.com/search", headers=headers, params=params)
    data = response.json()

    hits = data.get("response", {}).get("hits", [])
    if not hits:
        raise ValueError(f"No Genius results found for '{song_title}' by '{artist}'")

    result = hits[0]["result"]
    matched_artist = result.get("artist_names", "")
    if artist.lower() not in matched_artist.lower() and on_note is not None:
        on_note(f"Note: closest match was '{result['title']}' by '{matched_artist}' - verify this is the right song")

    song_url = result["url"]
    page = fetch_with_retry(song_url, headers={"User-Agent": headers["User-Agent"]})
    soup = BeautifulSoup(page.text, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})

    lyrics_parts = []
    for container in containers:
        for junk in container.find_all(attrs={"data-exclude-from-selection": "true"}):
            junk.decompose()
        lyrics_parts.append(container.get_text(separator="\n"))

    lyrics = "\n".join(lyrics_parts)
    if not lyrics.strip():
        raise ValueError(f"Lyrics came back empty for '{song_title}' by '{artist}' - check the title/artist spelling")
    return lyrics


_analyzer = SentimentIntensityAnalyzer()


def get_song_mood(lyrics):
    lines = [line for line in lyrics.split("\n") if line.strip()]
    line_scores = [_analyzer.polarity_scores(line)["compound"] for line in lines]
    compounds = [s for s in line_scores if abs(s) > 0.05]

    avg_mood = sum(compounds) / len(compounds) if compounds else 0
    mood_variance = statistics.variance(compounds) if len(compounds) > 1 else 0
    return round(avg_mood, 2), round(mood_variance, 2)


def nearest_category_by_mood(mood, variance):
    def distance(cat):
        target = CATEGORY_TARGETS[cat]
        return math.sqrt((target["mood"] - mood) ** 2 + (target["variance"] - variance) ** 2)
    return min(CATEGORY_TARGETS, key=distance)


def distance_similarity(mood, variance, category):
    target = CATEGORY_TARGETS[category]
    dist = math.sqrt((target["mood"] - mood) ** 2 + (target["variance"] - variance) ** 2)
    return 1 / (1 + dist)


def analyze_song_emotion(lyrics, mood=None):
    """Full emotion analysis (top/secondary emotion, confidence, 8-way profile).
    See emotion_engine.py for how this improves on a raw argmax over NRC counts."""
    valence = CALIBRATION.valence(mood) if mood is not None else None
    return analyze_lyrics(lyrics, valence)


def get_top_emotion(lyrics, mood=None):
    """Best-matching emotion for the lyrics, or None if no emotion words were found."""
    if not NRCLEX_AVAILABLE:
        return None
    try:
        return analyze_song_emotion(lyrics, mood)["top_emotion"]
    except Exception:
        return None


def fetch_itunes_track(title, artist):
    """
    Looks a song up on the iTunes Search API (free, no auth required) and returns
    {"duration_seconds", "preview_url", "track_url"}; any value may be None.
    """
    info = {"duration_seconds": None, "preview_url": None, "track_url": None}
    params = {"term": f"{title} {artist}", "media": "music", "entity": "song", "limit": 1}
    try:
        response = requests.get("https://itunes.apple.com/search", params=params, timeout=10)
        results = response.json().get("results", [])
        if not results:
            return info
        track = results[0]
        millis = track.get("trackTimeMillis")
        info["duration_seconds"] = round(millis / 1000) if millis else None
        info["preview_url"] = track.get("previewUrl")
        info["track_url"] = track.get("trackViewUrl")
        return info
    except (requests.exceptions.RequestException, ValueError):
        return info


def get_track_info(title, artist):
    """Returns (duration_seconds, preview_url); either may be None if unavailable."""
    info = fetch_itunes_track(title, artist)
    return info["duration_seconds"], info["preview_url"]


def format_duration(seconds):
    if seconds is None:
        return "-"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def make_song_record(title, artist, lyrics, with_track_info=True):
    """Score already-fetched lyrics and package them as a library entry.
    ``with_track_info=False`` skips the iTunes lookup (used by the offline benchmark)."""
    mood, variance = get_song_mood(lyrics)
    category = nearest_category_by_mood(mood, variance)
    emotion = analyze_song_emotion(lyrics, mood) if NRCLEX_AVAILABLE else None
    track = fetch_itunes_track(title, artist) if with_track_info else {
        "duration_seconds": None, "preview_url": None, "track_url": None}

    return {
        "title": title,
        "artist": artist,
        "lyrics": lyrics,
        "mood": mood,
        "variance": variance,
        "category": category,
        "top_emotion": (emotion or {}).get("top_emotion") or "-",
        "emotion_confidence": (emotion or {}).get("confidence"),
        "emotion_analysis": emotion,
        "word_count": len(lyrics.split()),
        "duration_seconds": track["duration_seconds"],
        "preview_url": track["preview_url"],
        "track_url": track["track_url"],
    }
