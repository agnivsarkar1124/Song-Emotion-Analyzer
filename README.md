# 🎵 Song Mood Explorer

An interactive dashboard that classifies songs by emotional category using their **actual lyrics** — not genre tags, not audio features, just the words themselves.

Enter a song and artist, and it fetches the lyrics, scores mood and emotional volatility, profiles discrete emotions, and predicts which of five mood categories the song belongs to — all visualized live. You can also **describe a mood in your own words** and get a ranked playlist, then **like, nope, save or skip** songs and watch a taste model learn what you actually want and re-rank the mix around it.

## What it does

1. **Fetches lyrics** via the Genius API (search + HTML scraping, since Genius doesn't expose lyrics directly through its API)
2. **Scores sentiment line-by-line** with VADER — capturing both:
   - **Mood**: average sentiment direction (positive vs. negative)
   - **Variance**: how much the song swings emotionally, line to line
3. **Profiles discrete emotions** using the NRC Emotion Lexicon (anger, joy, sadness, fear, trust, etc.) — not just positive/negative polarity
4. **Classifies the song** into one of five categories (Happy, Sad, Chill, Intense, Angry) using a combination of:
   - Statistical distance to data-calibrated mood/variance targets
   - TF-IDF content similarity to a reference set of songs per category

## Describe a mood, get a playlist

Type (or upload a `.txt` file with) any mood description — *"chill but slightly nostalgic for a rainy day"*, *"angry gym playlist"*, *"very romantic, not sad"* — and the app returns the **top K closest songs** from your library (plus the 15 seed songs, optionally) as a structured JSON playlist, inside the interactive mix described in the next section.

How retrieval works (`mood_search.py`):

1. **Query → mood space.** A small mood lexicon maps words to a valence/intensity position, an emotion mix and lyric keywords (query expansion: "rainy" → rain, storm, grey…). Modifiers scale the effect ("*slightly* nostalgic" counts for less than "*very* nostalgic") and negation flips it ("not sad").
2. **Every song lives in the same space:** standardised VADER mood and variance, an 8-way NRC emotion profile, and what its lyrics are *about*: transformer sentence embeddings (semantic search, below) plus a TF-IDF vector for exact keywords.
3. **Distance** = `1 − (w_emotion·cos(emotion profiles) + w_mood·exp(−d²/2σ²) over (valence, intensity) + w_theme·theme)`, with weights 0.30 / 0.35 / 0.35 and `theme = 0.75·semantic + 0.25·keyword` when semantic search is on (0.35 / 0.40 / 0.25 and keyword-only otherwise). The K smallest distances are the playlist; each track reports its component scores so you can see *why* it matched.
4. The retrieved lyrics also feed the existing **"Explain this playlist"** step, which asks Claude to explain the picks using only those excerpts.

The emotion and mood axes stay interpretable on purpose; the transformer handles meaning. The weights and the hand-written mood lexicon have not been tuned against labelled data.

### Semantic search (transformers)

With `sentence-transformers` installed (it's in `requirements-ml.txt`), `semantic.py` embeds every song's lyrics in two-line chunks with **all-MiniLM-L6-v2** and compares them with your description by cosine similarity. A song's semantic score blends its best-matching chunk with its top-3 average, so a query like *"missing someone who moved away"* finds a song about an empty room and a phone that never rings, even with no words in common. The card shows the lyric lines that matched best, and descriptions with no mood words at all (which keyword search used to reject) now work. Embeddings are computed once and cached as vectors in `.embedding_cache/` (gitignored), so searches stay fast. The model (~90 MB) downloads on first use. Without the library, or offline, the app falls back to keyword search; the header badge and sidebar say which is active. `SONG_MOOD_DISABLE_SEMANTIC=1` forces keywords, and `SONG_MOOD_EMBEDDING_MODEL` swaps the model.

Two engineering notes from making PyTorch and TensorFlow share one process: importing TensorFlow *before* the sentence-transformers stack segfaulted the whole app (reproduced on Linux with TF 2.21, torch 2.14, transformers 5), so `neural_ranker.py` preloads that stack first, and a subprocess regression test guards it. Streamlit's auto-reload watcher also trips over `transformers`' lazy imports, so `.streamlit/config.toml` turns it off; restart the app after editing code.

The emotion label itself is also better than before (`emotion_engine.py`). The old `argmax` over raw NRC counts frequently returned the lexicon's `positive` / `negative` polarity tags rather than an emotion, was biased toward whichever emotions tag the most words, counted negated words ("never happy") and counted a chorus every time it repeated. The new scorer drops polarity tags, normalises by each emotion's base rate (square-root-corrected, with smoothing), skips negated words, damps repeated lines, and nudges scores toward emotions that agree with the song's overall tone, then reports a confidence margin. Anger vs. fear vs. sadness is still the hard case for any word-lexicon method — check the confidence value.

## Personalised recommendations: a feedback loop that learns your taste

The mix is a two-way custom Streamlit component (`taste_ui.py`, built on `st.components.v2`) with six tabs:

| Tab | What you do there |
| --- | --- |
| 🎧 **For you** | Personalised ranking. ❤️ like / 👎 nope / ⭐ save / ⏭️ skip any card; the list re-ranks itself with animated moves, ▲/▼ badges show how far each song moved, and chips say why ("+ sad and energetic", "− lyrics about party & dance"). |
| 🃏 **Rate** | A swipe deck of the songs the model is *least sure about* (active learning: the ratings that teach it the most). Drag the card, or use ← → ↑ ↓. |
| 🧬 **Your taste** | Plain-language insights, an emotion × energy heat map, theme leanings, what your last rating changed, and a chart of how your profile evolved with every action. |
| ⚖️ **Compare** | Original (mood-only) vs personalised ranking as a slopegraph, plus overlap@K, Spearman rank correlation, songs added/dropped and why (with "undo nope"). |
| ⭐ **Saved** | Your saved songs, exportable as JSON. |
| { } **JSON** | The personalised playlist JSON. |

**What the model learns** (`preferences.py`, `themes.py`). Every song becomes a vector of standardised, readable features: tone (dark ↔ bright), energy (VADER line-to-line swing), its 8-way emotion mix, 13 lyrical themes (rain & weather, memories, heartbreak, night & city, …), and **interaction terms** emotion × energy. A like is +1, a nope −1, a save a stronger +1, a skip a weak −1 ("not now" ≠ "never"); your latest rating of a song replaces the earlier one. Weighted ridge regression fits a weight per feature in closed form after every click.

**"I like sad songs" vs "I like sad songs that are energetic".** The first is a positive weight on `sadness`; the second is a positive weight on `sadness × energy`, with little or no main-effect preference for sadness. Interaction terms carry a 3× larger ridge penalty, so the model only reaches for "sad *and* energetic" when your ratings actually require it (for example you like sad + energetic songs, nope sad + calm ones, *and* nope happy + energetic ones, so it isn't just "likes energy"). The emotion × energy grid in **Your taste** makes the difference visible, and `tests/test_preferences.py` checks both users end up with different profiles and rank the same unseen songs differently.

**Keras / TensorFlow: a Wide & Deep network** (`neural_ranker.py`). When TensorFlow (or JAX) is installed, a small Keras *Wide & Deep* model (Google's classic recommender architecture) trains on the same ratings: the *wide* part is a linear layer over every feature including the hand-made crosses (memorisation), and the *deep* part is a 16→8 ReLU MLP over the main features that can discover interactions nobody coded (generalisation). Its share of the taste score grows with your number of ratings (up to 60%), because a neural net trained on five ratings is mostly noise. The ridge weights stay the explanation layer, because they can be read directly. Training reuses one compiled model and resets its weights (~0.2 s per click instead of ~1.2 s for a rebuild), and predictions run as a NumPy forward pass that the tests check against Keras's own output. The sidebar's **"Is the neural net helping?"** button runs 5-fold cross-validation on *your* ratings and reports linear vs Wide & Deep accuracy, so you can see whether the network earns its place.

**Re-ranking.** `personal = (1 − a)·mood_match + a·(taste + 1)/2 − 0.04·skips`, where the personalisation weight `a` grows from 0 to at most 0.45 as you give feedback. Disliked songs are hidden from the personalised list. Feedback is saved to `taste_profile.json` next to `app.py` (titles and features only, no lyrics). You can download, load or reset it from the sidebar.

Honest caveats: the feature weights, penalties and blend schedules are reasonable defaults, not tuned values. With only a few dozen songs in the database, personalisation mostly re-orders a small pool. It shines once your library is bigger.

### Example playlist JSON

Abridged; every track also carries its emotion profile, tone/swing coordinates, preview and store links, and a short `why`.

```json
{
  "schema_version": 1,
  "query": "chill but slightly nostalgic for a rainy day",
  "k": 5,
  "database_size": 18,
  "weights": { "emotion": 0.35, "mood": 0.4, "theme": 0.25 },
  "parsed_mood": {
    "valence": -0.02, "intensity": -0.7, "description": "calm, neutral-toned",
    "concepts": [{ "term": "chill", "weight": 1.0 }, { "term": "nostalgic", "weight": 0.45 }, { "term": "rainy", "weight": 1.0 }]
  },
  "tracks": [
    {
      "rank": 1, "title": "…", "artist": "…",
      "distance": 0.23, "match_percent": 77,
      "scores": { "emotion": 0.67, "mood": 0.72, "theme": 1.0 },
      "top_emotion": "anticipation", "why": "Feels calm, neutral-toned; shares sadness & anticipation with your description; lyrics touch on rain, slow, window."
    }
  ]
}
```

## Try it

You need Python 3.9+ and two free keys: a **Genius API token** from [genius.com/api-clients](https://genius.com/api-clients) (required) and, optionally, an **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com) for the "Explain this playlist" button.

Run locally (a virtual environment keeps the dependencies tidy):
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # optional: Keras Wide & Deep model + semantic search
streamlit run app.py
```

`requirements-ml.txt` only installs what your machine supports. TensorFlow ships wheels for Apple Silicon, Linux and Windows on Python 3.10–3.13. On Python 3.14 it uses Keras on the JAX backend instead. **Intel Macs:** TensorFlow's last Intel-Mac release is 2.16.2 (Python 3.9–3.12), and there are no JAX wheels, so on Python 3.13+ nothing installs and the app runs on the linear taste model alone. For the neural model there, create the venv with Python 3.12 (`python3.12 -m venv venv`, e.g. after `brew install python@3.12`). The same file installs `sentence-transformers` (PyTorch) for semantic search: Apple Silicon on Python 3.10–3.14, Linux and Windows, and Intel Macs on Python ≤3.12 (PyTorch's last Intel-Mac release is 2.2.2). The sidebar shows which models are active. Set `SONG_MOOD_DISABLE_NEURAL=1` to force the linear model.

Then paste your Genius token into the sidebar, add some songs, and try a mood description. Genius blocks scraping from cloud hosting IPs, so lyric fetching only works when you run the app on your own machine.

## Benchmark: does personalisation actually help?

`benchmark.py` measures the feedback loop with **simulated listeners**: hand-written hidden tastes such as "likes sad songs only when they're energetic", plus a random-taste control. For each listener, mood query and seed it:

1. collects feedback through the app's own loop (the listener rates the top card of the Rate deck, the ranking updates, repeat);
2. after 10 and 25 ratings, ranks every song the listener hasn't rated with four systems, all on the same feedback: mood only (no personalisation), the linear taste model, the linear model with its emotion × energy interaction terms switched off (ablation), and linear + Keras Wide & Deep (if installed);
3. scores each ranking with **NDCG@5** and **precision@5** against the listener's true relevance (fits the mood × fits their taste).

```bash
python benchmark.py --token YOUR_GENIUS_TOKEN     # real lyrics for the songs in benchmark_songs.txt
python benchmark.py --synthetic                   # quick check that it runs (numbers not meaningful)
```

The first real run fetches ~65 songs' lyrics from Genius and caches them in `benchmark_cache.json` (gitignored, since lyrics are copyrighted), so later runs are offline and fast. It prints a results table, a per-listener breakdown and a headline line, and saves `benchmark_results.json`. If semantic search is installed, the mood ranking uses it; add `--semantic off` to benchmark with keyword search instead.

Read the numbers honestly: the listeners' tastes are hand-written over the same kinds of features the model sees, so this tests whether the loop *recovers a known taste*, not whether real people are happier. Quote results as "on simulated listeners". The random-taste control is there on purpose: personalisation can't help it and, with a 45% personalisation weight, actively hurts it, which is a real limitation (a confidence-aware personalisation weight would be the fix).

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app: tabs (Mood mix, Library, Preset moods), feedback handling, taste-profile sidebar, preset-category recommender, RAG explanation |
| `pipeline.py` | Streamlit-free lyric pipeline: Genius fetching, VADER mood/variance, emotion analysis, iTunes lookup, song records |
| `mood_search.py` | Free-text mood parsing, in-memory index, semantic-distance scoring of every song, JSON playlist |
| `emotion_engine.py` | Lyric emotion scoring (NRC lexicon with base-rate correction, negation, repeat damping, tone prior) |
| `themes.py` | 13 interpretable lyrical themes per song |
| `semantic.py` | Optional transformer semantic search: lyric chunking, cached embeddings, best-matching lines |
| `preferences.py` | Feedback events, taste features, ridge taste model, profile history, personalised re-ranking + comparison |
| `neural_ranker.py` | Optional Keras Wide & Deep taste network (TensorFlow or JAX backend) + cross-validation |
| `taste_ui.py` | The interactive two-way component (HTML/CSS/JS) and its view model |
| `benchmark.py`, `benchmark_songs.txt` | Simulated-listener benchmark (NDCG@5 / precision@5, ablation, control) and its song list |
| `.streamlit/config.toml` | Dark theme matching the mix |
| `tests/` | Unit tests using synthetic lyrics and synthetic listeners |

## The technical story

This project surfaced several real problems worth documenting, not just a feature list:

**Naive averaging hides emotional intensity.** Scoring a whole song's sentiment as one number crushed everything toward neutral — a song that's genuinely, consistently sad and a song that swings wildly between euphoric and devastated can average to the same score. Adding **variance** as a second axis (not just mood direction) was necessary to tell them apart.

**Category targets need to be calibrated against real data, not guessed.** Initial targets (e.g. "Happy" = 0.5 average mood) were set on an idealized -1 to 1 scale that real song lyrics almost never reach — most songs' line-level sentiment averages land much closer to neutral than intuition suggests. Recalibrating targets using percentiles of the actual fetched dataset (20th/50th/80th percentile of real mood/variance values) fixed a serious classification skew where most songs were defaulting into a single catch-all category.

**Sentiment polarity can't distinguish anger from sadness.** Both register as simply "negative" under VADER, so aggressive songs and heartbroken ballads scored similarly and were frequently misclassified against each other. Adding the **NRC Emotion Lexicon** as a second signal — which scores discrete emotions like anger and sadness independently, rather than a single positive/negative axis — addressed this directly.

**Infrastructure constraints shape technical decisions.** An initial attempt to use transformer-based sentence embeddings (`sentence-transformers`) hit a hard wall: no PyTorch wheel exists for Intel Mac + Python 3.14, and every version of the library depends on PyTorch. Rather than fighting the environment, the project fell back to TF-IDF for content similarity — a real example of adapting method choice to actual infrastructure rather than assuming the most sophisticated tool is always available. Moving development to a supported Python (Apple Silicon, Python 3.13) later made embeddings possible, so semantic search was added as an *optional* layer on top of TF-IDF rather than a replacement: machines that can't install PyTorch still get a working app.

**A wrong search result can silently corrupt a whole pipeline.** The Genius search API occasionally returns the wrong song (e.g. searching without enough context can return a completely different track), producing lyrics that look plausible but are wrong. This is now guarded against with an explicit artist-name check and a clear failure message instead of failing silently.

## Tech stack

- **Genius API** — lyrics retrieval
- **BeautifulSoup** — HTML parsing/scraping
- **VADER Sentiment** — line-level sentiment scoring
- **NRC Emotion Lexicon (NRCLex)** — discrete emotion profiling (scored by `emotion_engine.py`)
- **sentence-transformers (all-MiniLM-L6-v2, PyTorch), optional** — semantic lyric search with line-level matches
- **scikit-learn (TF-IDF, cosine similarity) + NumPy** — keyword matching, category matching and mood-search ranking
- **Streamlit** — app UI, with a custom two-way component (`st.components.v2`, vanilla JS + SVG charts)
- **Keras 3 + TensorFlow / JAX (optional)** — Wide & Deep neural taste model
- **NumPy ridge regression** — interpretable, always-on taste model with emotion × energy interactions
- **iTunes Search API** — track length, 30-second preview clips and store links (no key needed)
- **Anthropic API (optional)** — grounded explanation of recommendations (RAG generation step)

## Tests

```bash
pip install pytest
pytest
```

The tests use small synthetic lyrics and simulated listeners (no real songs) and cover the emotion scorer, query parsing, ranking, the taste model (including sad vs sad + energetic), feedback persistence, re-ranking and comparison, the component's view data, semantic search (with a stand-in encoder: paraphrase matching, caching, fallback), the benchmark, and, when Keras is installed, that the Wide & Deep network learns an interaction and that its NumPy forward pass matches Keras. `SONG_MOOD_TEST_REAL_MODEL=1 pytest` also runs a check with the real embedding model.

## Possible next steps

- Ground-truth evaluation: hand-label a small test set and measure actual classifier accuracy (and tune the retrieval weights and emotion priors against it)
- Evaluate semantic vs keyword search on a small hand-labelled query set (recall of the songs a person would pick)
- Feed compressed lyric embeddings into the taste model so it can learn "songs like these" beyond the 13 themes
- Tune the taste model's penalties and blend schedule against real feedback logs (the cross-validation button is the starting point)
- Time-of-day-aware recommendations (map current time to a target mood category automatically)
- Multi-language support (current pipeline assumes English-language lyrics and lexicons)
