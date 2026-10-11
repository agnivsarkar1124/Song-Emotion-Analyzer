import os

import numpy as np
import pytest

import neural_ranker
from emotion_engine import EMOTIONS
from preferences import (FEATURES, FeatureSpace, FeedbackEvent, PreferenceModel, ProfileStore,
                         latest_change, personalize, profile_history, song_states)
from themes import THEME_NAMES, theme_scores


def raw(emotion, tone, energy, themes=None):
    emotions = {e: 0.5 / (len(EMOTIONS) - 1) for e in EMOTIONS}
    emotions[emotion] = 0.5
    return {"tone": tone, "energy": energy, "emotions": emotions,
            "themes": {t: (themes or {}).get(t, 0.1) for t in THEME_NAMES}}


TONE = {"sadness": -0.9, "joy": 0.9, "anger": -0.7, "trust": 0.6, "fear": -0.6}
SONGS = {f"{e}-{lvl}": raw(e, TONE[e], energy)
         for e in TONE for lvl, energy in (("calm", -1.2), ("mid", 0.0), ("energetic", 1.2))}
SPACE = FeatureSpace(list(SONGS.values()))


def ev(action, key, features=None):
    return FeedbackEvent(action=action, key=key, title=key, artist="test",
                         features=features or SONGS[key], ts=0.0, id=key + action)


@pytest.fixture(autouse=True)
def linear_only(monkeypatch):
    """Most tests pin the deterministic linear model; neural tests opt back in."""
    monkeypatch.setattr(neural_ranker, "available", lambda: False)


def test_no_feedback_means_no_opinion():
    model = PreferenceModel([], SPACE)
    assert model.alpha == 0 and model.n_labeled == 0
    assert model.preference(SONGS["sadness-calm"]) == 0.0
    assert model.insights() == []


def test_likes_sad_songs_regardless_of_energy():
    events = [ev("like", "sadness-calm"), ev("like", "sadness-energetic"),
              ev("dislike", "joy-calm"), ev("dislike", "joy-energetic")]
    model = PreferenceModel(events, SPACE)
    matrix = model.emotion_energy_matrix()
    assert matrix["sadness"]["calm"] > 0.15 and matrix["sadness"]["energetic"] > 0.15
    assert abs(matrix["sadness"]["energetic"] - matrix["sadness"]["calm"]) < 0.2
    assert model.weight("emo:sadness") > 0
    assert abs(model.weight("emo:sadness*energy")) < abs(model.weight("emo:sadness"))
    assert any("sad songs, calm or energetic" in t for t in model.insights())


def test_likes_sad_songs_only_when_energetic():
    events = [ev("like", "sadness-energetic"), ev("save", "sadness-energetic"),
              ev("dislike", "sadness-calm"), ev("dislike", "joy-energetic"), ev("dislike", "joy-calm")]
    model = PreferenceModel(events, SPACE)
    matrix = model.emotion_energy_matrix()
    assert matrix["sadness"]["energetic"] > 0.15
    assert matrix["sadness"]["energetic"] - matrix["sadness"]["calm"] > 0.3
    assert model.weight("emo:sadness*energy") > 0.05
    assert matrix["joy"]["energetic"] < 0          # it's not just "likes energy"
    assert any("energetic" in t and "sad" in t.lower() for t in model.insights())
    # an unseen sad + steady song sits between the two rated extremes
    mid = model.preference(SONGS["sadness-mid"])
    assert model.preference(SONGS["sadness-calm"]) < mid < model.preference(SONGS["sadness-energetic"])


def test_the_two_users_rank_the_same_unseen_songs_differently():
    sad_any = PreferenceModel([ev("like", "sadness-calm"), ev("like", "sadness-energetic"),
                               ev("dislike", "joy-mid")], SPACE)
    sad_energetic = PreferenceModel([ev("like", "sadness-energetic"), ev("dislike", "sadness-calm"),
                                     ev("dislike", "joy-energetic")], SPACE)
    unseen_calm = raw("sadness", -0.8, -0.9)
    unseen_energetic = raw("sadness", -0.8, 0.9)
    assert sad_any.preference(unseen_calm) > 0
    assert sad_energetic.preference(unseen_energetic) > sad_energetic.preference(unseen_calm)
    assert sad_energetic.preference(unseen_calm) < sad_any.preference(unseen_calm)


def test_latest_rating_wins_and_saves_count_more():
    states = song_states([ev("like", "joy-calm"), ev("dislike", "joy-calm")])
    assert states["joy-calm"]["disliked"] and not states["joy-calm"]["liked"]
    liked = PreferenceModel([ev("like", "joy-calm"), ev("dislike", "anger-energetic")], SPACE)
    saved = PreferenceModel([ev("like", "joy-calm"), ev("save", "joy-calm"), ev("dislike", "anger-energetic")], SPACE)
    assert saved.preference(SONGS["joy-calm"]) > liked.preference(SONGS["joy-calm"])
    assert saved.alpha > liked.alpha


def test_skip_is_a_weak_negative():
    skip = PreferenceModel([ev("skip", "anger-energetic")], SPACE)
    dislike = PreferenceModel([ev("dislike", "anger-energetic")], SPACE)
    assert dislike.preference(SONGS["anger-energetic"]) < skip.preference(SONGS["anger-energetic"]) < 0


def test_profile_history_tracks_each_event():
    events = [ev("like", "sadness-energetic"), ev("dislike", "sadness-calm"), ev("dislike", "joy-energetic")]
    history = profile_history(events, list(SONGS.values()))
    assert [h["step"] for h in history] == [1, 2, 3]
    assert set(history[-1]["weights"]) == set(FEATURES)
    assert history[0]["strength"] < history[-1]["strength"]
    changes = latest_change(history)
    assert changes and all(c["delta"] != 0 for c in changes)


def test_profile_store_round_trip(tmp_path):
    store = ProfileStore(str(tmp_path / "taste.json"))
    assert store.load() == []
    events = [ev("like", "joy-calm"), ev("skip", "fear-mid")]
    store.save(events)
    loaded = store.load()
    assert [(e.action, e.key) for e in loaded] == [("like", "joy-calm"), ("skip", "fear-mid")]
    (tmp_path / "taste.json").write_text("{not json")
    assert store.load() == []


def test_theme_scores_are_bounded_and_ignore_section_markers():
    scores = theme_scores("[Chorus]\nrain rain on the window, I remember yesterday")
    assert set(scores) == set(THEME_NAMES)
    assert scores["rain & weather"] > scores["party & dance"] == 0.0
    assert all(0 <= v <= 1 for v in scores.values())
    assert theme_scores("")["memories"] == 0.0


# --- end to end over the real search index -------------------------------------------

def test_personalize_reranks_and_compares(monkeypatch):
    from mood_search import Calibration, MoodIndex
    from tests.helpers import TARGETS, make_library

    index = MoodIndex(make_library(), Calibration.from_targets(TARGETS))
    sq = index.score_all("something moody")
    plain = personalize(index, sq, [], k=5)
    assert plain["alpha"] == 0
    assert [t["key"] for t in plain["personal"]] == [t["key"] for t in plain["base"]]

    tracks = {t["key"]: t for t in plain["base"] + plain["deck"]}
    first, second = plain["base"][0], plain["base"][1]
    events = [FeedbackEvent.create("dislike", first), FeedbackEvent.create("like", second),
              FeedbackEvent.create("save", second)]
    result = personalize(index, sq, events, k=5)
    keys = [t["key"] for t in result["personal"]]
    assert first["key"] not in keys                          # disliked songs are hidden
    assert keys[0] == second["key"]                          # liked + saved song rises to the top
    assert result["comparison"]["excluded"] == 1
    assert any(d["key"] == first["key"] for d in result["comparison"]["dropped"])
    assert result["personal"][0]["state"]["saved"]
    assert all(t["key"] not in {first["key"], second["key"]} for t in result["deck"])
    assert 0 < result["alpha"] < 0.45
    assert tracks  # deck + base cover the songs we rated


# --- neural model (only when Keras + a backend are installed) ---------------------------

def _neural_installed():
    return neural_ranker._load_keras() is not None


@pytest.mark.skipif(os.environ.get("SONG_MOOD_DISABLE_NEURAL") == "1", reason="neural disabled")
def test_neural_model_learns_an_interaction(monkeypatch):
    monkeypatch.undo()
    if not _neural_installed():
        pytest.skip("Keras with a trainable backend is not installed")
    rng = np.random.default_rng(0)
    raws, events = [], []
    for i in range(40):
        emotion = rng.choice(["sadness", "joy"])
        energy = float(rng.uniform(-1.5, 1.5))
        r = raw(emotion, TONE[emotion], energy)
        raws.append(r)
        liked = emotion == "sadness" and energy > 0.3
        events.append(FeedbackEvent(action="like" if liked else "dislike", key=f"s{i}", title=f"s{i}",
                                    artist="t", features=r, ts=0.0, id=str(i)))
    space = FeatureSpace(raws)
    model = PreferenceModel(events, space, neural=True)
    assert model.neural_status == "active" and 0 < model.beta <= 0.6
    scores = model.preferences([raw("sadness", -0.9, 1.2), raw("sadness", -0.9, -1.2), raw("joy", 0.9, 1.2)])
    assert scores["neural"][0] > scores["neural"][1]
    assert scores["neural"][0] > scores["neural"][2]
    check = model.check()
    assert check and 0 <= check["neural_accuracy"] <= 1 and 0 <= check["linear_accuracy"] <= 1


def test_numpy_forward_pass_matches_keras(monkeypatch):
    monkeypatch.undo()
    if not _neural_installed():
        pytest.skip("Keras with a trainable backend is not installed")
    rng = np.random.default_rng(3)
    Xw, Xd = rng.normal(size=(12, 6)), rng.normal(size=(12, 4))
    y = np.where(rng.normal(size=12) > 0, 1.0, -1.0)
    y[:2] = [1.0, -1.0]
    model = neural_ranker.WideDeepModel.train(Xw, Xd, y, np.ones(12))
    keras_out = neural_ranker._trainers[(6, 4)].predict(Xw, Xd)
    assert np.allclose(model.predict(Xw, Xd), keras_out, atol=1e-4)
    again = neural_ranker.WideDeepModel.train(Xw, Xd, y, np.ones(12))      # reused trainer is reset
    assert np.allclose(again.predict(Xw, Xd), model.predict(Xw, Xd), atol=1e-4)
