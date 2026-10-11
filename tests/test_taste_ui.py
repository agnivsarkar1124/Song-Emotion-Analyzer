import json

import pytest

import neural_ranker
import taste_ui
from mood_search import Calibration, MoodIndex
from preferences import FeedbackEvent, latest_change, personalize, profile_history
from tests.helpers import TARGETS, make_library


@pytest.fixture()
def mix(monkeypatch):
    monkeypatch.setattr(neural_ranker, "available", lambda: False)
    library = make_library()
    library[0]["title"] = "Window Seat Rain </script><img src=x onerror=alert(1)>"
    index = MoodIndex(library, Calibration.from_targets(TARGETS))
    sq = index.score_all("rainy and nostalgic")
    first = personalize(index, sq, [], k=4)
    events = [FeedbackEvent.create("like", first["base"][1]), FeedbackEvent.create("save", first["base"][1]),
              FeedbackEvent.create("dislike", first["base"][0])]
    result = personalize(index, sq, events, k=4)
    history = profile_history(events, result["population"])
    playlist = index.playlist(sq, result["personal"])
    songs = {s["key"]: s["src"] for s in index.songs}
    view = taste_ui.build_view(result, playlist, history, latest_change(history),
                               taste_ui.saved_items(result["model"].states, songs),
                               {"id": "x", "text": "hi", "detail": None}, version=len(events),
                               engine=neural_ranker.backend_info())
    return view, events


def test_view_is_plain_json_without_lyrics(mix):
    view, _ = mix
    text = json.dumps(view)
    assert json.loads(text)["version"] == 3
    assert "Old photographs in the drawer" not in text            # no lyrics leak to the browser
    assert '"lyrics"' not in text
    for t in view["personal"] + view["base"] + view["deck"]:
        assert set(t) >= {"key", "title", "state", "rank_change", "preference", "top_themes"}


def test_view_reflects_feedback(mix):
    view, events = mix
    liked_key, disliked_key = events[0].key, events[2].key
    assert view["personal"][0]["key"] == liked_key and view["personal"][0]["state"]["saved"]
    assert disliked_key not in {t["key"] for t in view["personal"]}
    assert [s["key"] for s in view["saved"]] == [liked_key]
    assert len(view["chart"]["steps"]) == 3 and view["chart"]["series"]
    assert view["profile"]["n_rated"] == 2 and "raw_weights" not in view["profile"]
    assert view["comparison"]["excluded"] == 1


def test_saved_items_only_keep_https_links():
    states = {"a|b": {"saved": True, "title": "A", "artist": "B"}, "c|d": {"saved": False}}
    songs = {"a|b": {"track_url": "javascript:alert(1)", "top_emotion": "-"}}
    assert taste_ui.saved_items(states, songs) == [
        {"key": "a|b", "title": "A", "artist": "B", "top_emotion": None, "track_url": None}]
