"""Shared helpers for the tests: build song dicts the same way app.add_song_to_library does."""
import statistics

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tests.fixtures import SONGS

TARGETS = {
    "Happy":   {"mood": 0.33, "variance": 0.15},
    "Sad":     {"mood": -0.07, "variance": 0.15},
    "Chill":   {"mood": 0.17, "variance": 0.10},
    "Intense": {"mood": 0.17, "variance": 0.23},
    "Angry":   {"mood": -0.07, "variance": 0.23},
}

_an = SentimentIntensityAnalyzer()


def mood_and_variance(lyrics):
    lines = [line for line in lyrics.split("\n") if line.strip()]
    scores = [_an.polarity_scores(line)["compound"] for line in lines]
    compounds = [s for s in scores if abs(s) > 0.05]
    mood = sum(compounds) / len(compounds) if compounds else 0
    var = statistics.variance(compounds) if len(compounds) > 1 else 0
    return round(mood, 2), round(var, 2)


def make_library():
    library = []
    for title, (artist, lyrics) in SONGS.items():
        mood, var = mood_and_variance(lyrics)
        library.append({"title": title, "artist": artist, "lyrics": lyrics, "mood": mood,
                        "variance": var, "category": "Chill", "top_emotion": "-"})
    return library
