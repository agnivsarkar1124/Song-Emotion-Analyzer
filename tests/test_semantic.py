import os
import re

import numpy as np
import pytest

import semantic
from mood_search import Calibration, MoodIndex
from tests.helpers import TARGETS, make_library

# A stand-in "embedding model": words map to shared concept dimensions, so paraphrases
# with no words in common still land close together (what a real model does, crudely).
CONCEPTS = {
    "leave": {"left", "leave", "leaving", "gone", "goodbye", "away", "went", "moved"},
    "alone": {"alone", "lonely", "nobody", "empty", "myself", "silence"},
    "rain": {"rain", "storm", "clouds", "grey", "drizzle", "window"},
    "dance": {"dance", "party", "jump", "celebrate", "dancing"},
    "fear": {"afraid", "scared", "shadows", "nervous", "dark"},
    "love": {"love", "kiss", "darling", "hold", "tender"},
}
DIM = 64


def fake_encoder(calls=None):
    def encode(texts):
        if calls is not None:
            calls.append(len(texts))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z']+", text.lower()):
                for c, (name, words) in enumerate(CONCEPTS.items()):
                    if word in words:
                        out[i, c] += 1.0
                out[i, len(CONCEPTS) + hash(word) % (DIM - len(CONCEPTS))] += 0.05
        return out
    return encode


@pytest.fixture()
def fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "CACHE_DIR", str(tmp_path / "cache"))
    calls = []
    semantic.set_encoder(fake_encoder(calls), name="fake-model")
    return calls


def test_chunks_skip_markers_and_repeats():
    chunks = semantic.chunk_lyrics("[Chorus]\nHold on\nHold on\nLet it go\n\nhold on\nNever again")
    assert chunks == ["Hold on / Let it go", "Let it go / Never again"]
    assert semantic.chunk_lyrics("") == []
    assert semantic.chunk_lyrics("Only one line") == ["Only one line"]


def test_disabled_encoder_means_unavailable():
    assert not semantic.available()
    assert semantic.status()["active"] is False


def test_query_without_any_mood_words_now_works(fake_model):
    index = MoodIndex(make_library(), Calibration.from_targets(TARGETS))
    assert index.semantic is not None
    playlist, _ = index.search("someone went away and now it is just me by myself", k=3)
    top = playlist["tracks"][0]
    assert top["title"] == "Cold Telephone"
    assert top["matched_line"] and re.search(r"left|goodbye|alone|empty|nobody", top["matched_line"], re.I)
    assert top["scores"]["semantic"] > 0.15
    assert playlist["search_engine"] == {"semantic": True, "model": "fake-model"}


def test_without_semantic_the_same_query_is_rejected():
    index = MoodIndex(make_library(), Calibration.from_targets(TARGETS))
    with pytest.raises(ValueError):
        index.search("someone went away and now it is just me by myself")


def test_embeddings_are_cached_on_disk(fake_model):
    library = make_library()
    MoodIndex(library, Calibration.from_targets(TARGETS))
    first = sum(fake_model)
    fake_model.clear()
    index = MoodIndex(library, Calibration.from_targets(TARGETS))
    assert first > 0 and sum(fake_model) == 0          # nothing re-encoded the second time
    index.search("rain on the window", k=2)
    assert sum(fake_model) == 1                         # only the query


def test_broken_model_falls_back_to_keywords(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "CACHE_DIR", str(tmp_path))

    def broken(texts):
        raise RuntimeError("model exploded")

    semantic.set_encoder(broken, name="broken")
    index = MoodIndex(make_library(), Calibration.from_targets(TARGETS))
    assert index.semantic is None and "exploded" in index.semantic_error
    playlist, _ = index.search("rainy and nostalgic", k=2)
    assert playlist["search_engine"]["semantic"] is False
    assert all(t["matched_line"] is None for t in playlist["tracks"])


@pytest.mark.skipif(os.environ.get("SONG_MOOD_TEST_REAL_MODEL") != "1",
                    reason="set SONG_MOOD_TEST_REAL_MODEL=1 to test the real model (downloads ~90 MB once)")
def test_real_model_matches_a_paraphrase(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(semantic, "CACHE_DIR", str(tmp_path))
    semantic.set_encoder(None)
    if not semantic.available():
        pytest.skip(semantic.status()["label"])
    index = MoodIndex(make_library(), Calibration.from_targets(TARGETS))
    playlist, _ = index.search("my partner walked out and the house feels empty", k=2)
    assert "Cold Telephone" in [t["title"] for t in playlist["tracks"]]


def test_tensorflow_then_sentence_transformers_does_not_crash():
    """Regression: importing TensorFlow before the sentence-transformers stack segfaulted the
    whole process. neural_ranker now preloads that stack first. Runs in a subprocess."""
    import importlib.util
    import subprocess
    import sys
    if not (importlib.util.find_spec("sentence_transformers") and importlib.util.find_spec("tensorflow")):
        pytest.skip("needs both TensorFlow and sentence-transformers installed")
    code = ("import neural_ranker; assert neural_ranker.available(); "
            "import sentence_transformers, transformers, tokenizers; print('ok')")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-2000:]
