"""
Optional transformer-based semantic search over lyrics (sentence-transformers).

Keyword matching (TF-IDF + a hand-written mood lexicon) only finds songs that use
the words you typed or their listed synonyms. A sentence-embedding model maps text
to vectors where *meaning* is close, so "missing someone who moved away" can match
a lyric about an empty room and a phone that never rings.

How it works:
  * each song's lyrics are split into short chunks (two consecutive lines, repeats
    removed) and every chunk is embedded once (cached on disk as vectors only);
  * the mood description is embedded at query time;
  * a song's semantic score blends its best-matching chunk with its top-3 average
    (cosine similarity), calibrated to [0, 1];
  * the best-matching chunk is returned, so the UI can show *which line* matched.

Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (~90 MB, downloaded on
first use, runs fine on a laptop CPU). Override with ``SONG_MOOD_EMBEDDING_MODEL``.
If sentence-transformers isn't installed, the model can't be downloaded, or
``SONG_MOOD_DISABLE_SEMANTIC=1`` is set, ``available()`` is False and the app
falls back to keyword matching.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import threading

import numpy as np

MODEL_NAME = os.environ.get("SONG_MOOD_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_DIR = os.environ.get("SONG_MOOD_EMBEDDING_CACHE",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), ".embedding_cache"))
LINES_PER_CHUNK = 2
MAX_CHUNKS = 120          # per song; long songs are sampled evenly
TOP_CHUNKS = 3
COS_LOW, COS_HIGH = 0.10, 0.55   # cosine <= LOW -> 0, >= HIGH -> 1 (typical MiniLM range for related text)

_SECTION_RE = re.compile(r"^\[.*\]$")
_lock = threading.RLock()
_state = {"encoder": None, "name": None, "error": None}   # encoder: callable, or False once known unavailable


# --- encoder ------------------------------------------------------------------------------

def _disabled() -> bool:
    return os.environ.get("SONG_MOOD_DISABLE_SEMANTIC", "").strip() not in ("", "0")


def installed() -> bool:
    """Cheap check (no model load): is sentence-transformers importable?"""
    return not _disabled() and importlib.util.find_spec("sentence_transformers") is not None


def set_encoder(fn, name: str = "custom") -> None:
    """Install a custom encoder ``fn(list[str]) -> array (n, d)`` (tests, other models).
    ``fn=None`` resets to the default model; ``fn=False`` disables semantic search."""
    with _lock:
        _state.update(encoder=fn, name=name if fn else None, error=None)


def _load():
    with _lock:
        if _state["encoder"] is not None:
            return _state["encoder"] or None
        if _disabled():
            _state.update(encoder=False, error="disabled by SONG_MOOD_DISABLE_SEMANTIC")
            return None
        if not installed():
            _state.update(encoder=False, error="sentence-transformers is not installed")
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME)

            def encode(texts):
                return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)

            _state.update(encoder=encode, name=MODEL_NAME, error=None)
        except Exception as e:          # offline, blocked download, broken torch install ...
            _state.update(encoder=False, error=f"could not load {MODEL_NAME}: {e}")
        return _state["encoder"] or None


def available() -> bool:
    """True if a working encoder is ready (loads - and on first ever use downloads - the model)."""
    return _load() is not None


def status() -> dict:
    """For the UI, without forcing a model load."""
    encoder = _state["encoder"]
    if encoder:
        return {"active": True, "label": f"semantic ({_short(_state['name'])})"}
    if encoder is False:
        return {"active": False, "label": f"keywords only - {_state['error']}"}
    if installed():
        return {"active": False, "label": f"semantic ({_short(MODEL_NAME)}) - loads on first search"}
    return {"active": False, "label": "keywords only - install sentence-transformers for semantic search"}


def model_name() -> str | None:
    return _state["name"] if _state["encoder"] else None


def _short(name) -> str:
    return str(name or "").split("/")[-1]


def encode(texts: list) -> np.ndarray:
    """Unit-normalised float32 embeddings, shape (len(texts), d)."""
    encoder = _load()
    if encoder is None:
        raise RuntimeError(_state["error"] or "semantic search unavailable")
    vectors = np.asarray(encoder(list(texts)), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1.0)


# --- lyrics -> chunks ---------------------------------------------------------------------------

def chunk_lyrics(lyrics: str) -> list:
    """Two-line chunks of the lyrics, with section markers and repeated lines removed."""
    seen, lines = set(), []
    for raw in (lyrics or "").replace("’", "'").split("\n"):
        line = raw.strip()
        key = line.lower()
        if len(line) < 3 or _SECTION_RE.match(line) or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    if not lines:
        return []
    if len(lines) <= LINES_PER_CHUNK:
        return [" / ".join(lines)]
    chunks = [" / ".join(lines[i:i + LINES_PER_CHUNK]) for i in range(len(lines) - LINES_PER_CHUNK + 1)]
    if len(chunks) > MAX_CHUNKS:
        keep = np.linspace(0, len(chunks) - 1, MAX_CHUNKS).round().astype(int)
        chunks = [chunks[i] for i in keep]
    return chunks


def _cache_path(lyrics: str) -> str:
    digest = hashlib.sha1(f"{_state['name']}|{LINES_PER_CHUNK}|{MAX_CHUNKS}|{lyrics}".encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, digest + ".npy")


class LyricEmbeddings:
    """Chunk embeddings for a list of songs' lyrics, aligned with that list."""

    def __init__(self, lyrics_list: list, use_disk_cache: bool = True):
        self.chunks = [chunk_lyrics(lyrics) for lyrics in lyrics_list]
        self.matrices: list = [None] * len(lyrics_list)
        todo = []
        for i, lyrics in enumerate(lyrics_list):
            if not self.chunks[i]:
                continue
            if use_disk_cache:
                try:
                    cached = np.load(_cache_path(lyrics))
                    if cached.shape[0] == len(self.chunks[i]):
                        self.matrices[i] = cached
                        continue
                except (OSError, ValueError):
                    pass
            todo.append(i)
        if todo:
            flat = [chunk for i in todo for chunk in self.chunks[i]]
            vectors = encode(flat)
            start = 0
            for i in todo:
                n = len(self.chunks[i])
                self.matrices[i] = vectors[start:start + n]
                start += n
                if use_disk_cache:
                    try:
                        os.makedirs(CACHE_DIR, exist_ok=True)
                        np.save(_cache_path(lyrics_list[i]), self.matrices[i])
                    except OSError:
                        pass

    def score(self, query_text: str):
        """(scores in [0, 1] per song, best-matching chunk per song or None)."""
        q = encode([query_text])[0]
        scores = np.zeros(len(self.matrices))
        best = [None] * len(self.matrices)
        for i, matrix in enumerate(self.matrices):
            if matrix is None or not len(matrix):
                continue
            sims = matrix @ q
            top = np.sort(sims)[::-1][:TOP_CHUNKS]
            agg = 0.6 * float(top[0]) + 0.4 * float(np.mean(top))
            scores[i] = min(1.0, max(0.0, (agg - COS_LOW) / (COS_HIGH - COS_LOW)))
            best[i] = self.chunks[i][int(np.argmax(sims))]
        return scores, best
