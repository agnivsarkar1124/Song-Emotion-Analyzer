import pytest

from emotion_engine import EMOTIONS, analyze_lyrics, legacy_top_emotion, lexicon_available
from tests.fixtures import SONGS

pytestmark = pytest.mark.skipif(not lexicon_available(), reason="nrclex lexicon not installed")

POLARITY_TAGS = {"positive", "negative"}


def test_never_returns_a_polarity_tag_as_the_emotion():
    for title, (_, lyrics) in SONGS.items():
        assert analyze_lyrics(lyrics)["top_emotion"] in EMOTIONS, title


def test_legacy_method_did_return_polarity_tags():
    # Documents the bug being fixed: argmax over raw counts often picks "positive"/"negative".
    legacy = [legacy_top_emotion(lyrics) for _, lyrics in SONGS.values()]
    assert any(tag in POLARITY_TAGS for tag in legacy)


def test_profile_is_a_distribution():
    result = analyze_lyrics(SONGS["Cold Telephone"][1], valence=-1.0)
    assert set(result["profile"]) == set(EMOTIONS)
    assert sum(result["profile"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["top_emotion"] != result["secondary_emotion"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_sad_and_joyful_songs_are_told_apart():
    assert analyze_lyrics(SONGS["Cold Telephone"][1], valence=-1.0)["top_emotion"] == "sadness"
    assert analyze_lyrics(SONGS["Fireworks Forever"][1], valence=1.5)["top_emotion"] == "joy"


def test_no_lexicon_words_gives_none():
    result = analyze_lyrics("xyzzy plugh quux")
    assert result["top_emotion"] is None and result["confidence"] == 0.0
    assert analyze_lyrics("")["top_emotion"] is None


def test_section_markers_are_ignored():
    assert analyze_lyrics("[Chorus]\n[Verse 1]")["top_emotion"] is None


def test_negated_emotion_words_are_not_counted():
    plain = analyze_lyrics("happy happy happy")
    negated = analyze_lyrics("not happy, never happy, no happy")
    assert plain["top_emotion"] == "joy"
    assert negated["top_emotion"] is None and negated["negated_hits"] == 3


def test_repeated_hooks_are_damped():
    text = "\n".join(["cry cry tears"] * 8 + ["dance and laugh with joy"])
    damped = analyze_lyrics(text)["profile"]["joy"]
    undamped = analyze_lyrics(text, repeat_decay=0.0)["profile"]["joy"]
    assert damped > undamped


def test_tone_prior_moves_scores_in_the_expected_direction():
    text = "love and fear and hope and cry"
    bright = analyze_lyrics(text, valence=1.5)["profile"]
    dark = analyze_lyrics(text, valence=-1.5)["profile"]
    assert bright["joy"] > dark["joy"]
    assert dark["sadness"] > bright["sadness"]


def test_inflected_forms_are_found():
    # "loved" / "tears" are not in the lexicon verbatim but their stems are
    assert analyze_lyrics("loved loved loved")["coverage"] > 0
