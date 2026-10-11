import json

import pytest

from emotion_engine import lexicon_available
from mood_search import Calibration, MoodIndex, parse_mood_description, playlist_to_json
from tests.helpers import TARGETS, make_library

pytestmark = pytest.mark.skipif(not lexicon_available(), reason="nrclex lexicon not installed")

CAL = Calibration.from_targets(TARGETS)


@pytest.fixture(scope="module")
def index():
    return MoodIndex(make_library(), CAL)


def test_calibration_centres_the_typical_song():
    assert CAL.mood_center == pytest.approx(0.17)
    assert CAL.var_center == pytest.approx(0.15)
    assert CAL.valence(0.17) == 0 and CAL.valence(0.33) == pytest.approx(0.8)
    assert CAL.valence(10) == 1.5 and CAL.intensity(-10) == -1.5     # clipped


def test_parse_example_from_the_brief():
    q = parse_mood_description("chill but slightly nostalgic for a rainy day")
    weights = {c["term"]: c["weight"] for c in q.concepts}
    assert set(weights) == {"chill", "nostalgic", "rainy"}
    assert weights["nostalgic"] < weights["chill"]          # "slightly" down-weights it
    assert q.intensity < -0.4                                # chill dominates: calm
    assert q.emotions["sadness"] > 0 and q.emotions["trust"] > 0
    assert sum(q.emotions.values()) == pytest.approx(1.0, abs=1e-3)
    assert "rain" in q.keywords and "remember" in q.keywords


def test_modifiers_scale_magnitude():
    slight = parse_mood_description("slightly sad")
    plain = parse_mood_description("sad")
    very = parse_mood_description("very sad")
    assert abs(slight.valence) < abs(plain.valence) < abs(very.valence)


def test_negation_flips_direction():
    assert parse_mood_description("sad").valence < 0 < parse_mood_description("not sad").valence


def test_empty_and_unintelligible_descriptions():
    with pytest.raises(ValueError):
        parse_mood_description("   ")


def test_unintelligible_description_raises_in_search(index):
    with pytest.raises(ValueError):
        index.search("zzzz qqqq")


def test_top_k_is_sorted_by_distance_and_capped(index):
    playlist, _ = index.search("calm and peaceful", k=4)
    distances = [t["distance"] for t in playlist["tracks"]]
    assert len(distances) == 4 and distances == sorted(distances)
    assert all(0.0 <= d <= 1.0 for d in distances)
    assert [t["rank"] for t in playlist["tracks"]] == [1, 2, 3, 4]
    assert len(index.search("calm", k=999)[0]["tracks"]) == len(index)
    assert len(index.search("calm", k=0)[0]["tracks"]) == 1


def test_search_is_deterministic(index):
    a, _ = index.search("angry", k=5)
    b, _ = index.search("angry", k=5)
    assert [t["title"] for t in a["tracks"]] == [t["title"] for t in b["tracks"]]


@pytest.mark.parametrize("query, expected", [
    ("a rainy day, nostalgic", "Window Seat Rain"),
    ("furious and aggressive", "Burn The Ladder"),
    ("spooky and anxious", "Midnight Static"),
    ("lonely heartbreak", "Cold Telephone"),
    ("romantic and tender", "Slow Dance In The Kitchen"),
])
def test_obvious_queries_rank_the_obvious_song_first(index, query, expected):
    playlist, _ = index.search(query, k=3)
    assert playlist["tracks"][0]["title"] == expected


def test_playlist_json_schema_and_no_lyrics_leak(index):
    playlist, hits = index.search("chill but slightly nostalgic for a rainy day", k=5)
    text = playlist_to_json(playlist)
    assert json.loads(text) == playlist
    assert playlist["schema_version"] == 1 and playlist["k"] == 5
    for key in ("query", "weights", "parsed_mood", "summary", "tracks", "generated_at", "database_size"):
        assert key in playlist
    track = playlist["tracks"][0]
    for key in ("rank", "title", "artist", "distance", "match_percent", "scores", "emotion_profile",
                "valence", "intensity", "why", "preview_url"):
        assert key in track
    assert "lyrics" not in track and "Old photographs in the drawer" not in text
    assert all("lyrics" in h for h in hits)            # but the RAG step still gets them
    assert sum(playlist["weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_non_https_urls_are_dropped():
    library = make_library()
    library[0]["preview_url"] = "javascript:alert(1)"
    library[0]["track_url"] = "http://insecure.example.com"
    playlist, _ = MoodIndex(library, CAL).search("rainy", k=len(library))
    for t in playlist["tracks"]:
        assert t["preview_url"] is None and t["track_url"] is None


def test_empty_database_raises():
    with pytest.raises(ValueError):
        MoodIndex([], CAL).search("calm")
