# 🎵 Song Mood Explorer

An interactive dashboard that classifies songs by emotional category using their **actual lyrics** — not genre tags, not audio features, just the words themselves.

Enter a song and artist, and it fetches the lyrics, scores mood and emotional volatility, profiles discrete emotions, and predicts which of five mood categories the song belongs to — all visualized live.

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

Type (or upload a `.txt` file with) any mood description — *"chill but slightly nostalgic for a rainy day"*, *"angry gym playlist"*, *"very romantic, not sad"* — and the app returns the **top K closest songs** from your library (plus the 15 seed songs, optionally) as a structured JSON playlist and an interactive view with four tabs: **Playlist** (colour-coded, scrollable, click a card for score breakdown, preview clip and search links), **Mood map** (valence × intensity scatter; click a dot to jump to its card), **Breakdown** (how your words were interpreted), and **JSON** (copy / download).

How retrieval works (`mood_search.py`):

1. **Query → mood space.** A small mood lexicon maps words to a valence/intensity position, an emotion mix and lyric keywords (query expansion: "rainy" → rain, storm, grey…). Modifiers scale the effect ("*slightly* nostalgic" counts for less than "*very* nostalgic") and negation flips it ("not sad").
2. **Every song lives in the same space:** standardised VADER mood and variance, an 8-way NRC emotion profile, and a TF-IDF vector over its lyrics.
3. **Distance** = `1 − (0.35·cos(emotion profiles) + 0.40·exp(−d²/2σ²) over (valence, intensity) + 0.25·tanh(cos(TF-IDF)/0.12))`. The K smallest distances are the playlist; each track reports its three component scores so you can see *why* it matched.
4. The retrieved lyrics also feed the existing **"Explain this playlist"** step, which asks Claude to explain the picks using only those excerpts.

This is an interpretable hybrid, **not a neural embedding** (see the PyTorch note below). Its weights and the hand-written mood lexicon have not been tuned against labelled data.

The emotion label itself is also better than before (`emotion_engine.py`). The old `argmax` over raw NRC counts frequently returned the lexicon's `positive` / `negative` polarity tags rather than an emotion, was biased toward whichever emotions tag the most words, counted negated words ("never happy") and counted a chorus every time it repeated. The new scorer drops polarity tags, normalises by each emotion's base rate (square-root-corrected, with smoothing), skips negated words, damps repeated lines, and nudges scores toward emotions that agree with the song's overall tone, then reports a confidence margin. Anger vs. fear vs. sadness is still the hard case for any word-lexicon method — check the confidence value.

## Try it

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Genius API token from [genius.com/api-clients](https://genius.com/api-clients).

## The technical story

This project surfaced several real problems worth documenting, not just a feature list:

**Naive averaging hides emotional intensity.** Scoring a whole song's sentiment as one number crushed everything toward neutral — a song that's genuinely, consistently sad and a song that swings wildly between euphoric and devastated can average to the same score. Adding **variance** as a second axis (not just mood direction) was necessary to tell them apart.

**Category targets need to be calibrated against real data, not guessed.** Initial targets (e.g. "Happy" = 0.5 average mood) were set on an idealized -1 to 1 scale that real song lyrics almost never reach — most songs' line-level sentiment averages land much closer to neutral than intuition suggests. Recalibrating targets using percentiles of the actual fetched dataset (20th/50th/80th percentile of real mood/variance values) fixed a serious classification skew where most songs were defaulting into a single catch-all category.

**Sentiment polarity can't distinguish anger from sadness.** Both register as simply "negative" under VADER, so aggressive songs and heartbroken ballads scored similarly and were frequently misclassified against each other. Adding the **NRC Emotion Lexicon** as a second signal — which scores discrete emotions like anger and sadness independently, rather than a single positive/negative axis — addressed this directly.

**Infrastructure constraints shape technical decisions.** An initial attempt to use transformer-based sentence embeddings (`sentence-transformers`) hit a hard wall: no PyTorch wheel exists for Intel Mac + Python 3.14, and every version of the library depends on PyTorch. Rather than fighting the environment, the project fell back to TF-IDF for content similarity — a real example of adapting method choice to actual infrastructure rather than assuming the most sophisticated tool is always available.

**A wrong search result can silently corrupt a whole pipeline.** The Genius search API occasionally returns the wrong song (e.g. searching without enough context can return a completely different track), producing lyrics that look plausible but are wrong. This is now guarded against with an explicit artist-name check and a clear failure message instead of failing silently.

## Tech stack

- **Genius API** — lyrics retrieval
- **BeautifulSoup** — HTML parsing/scraping
- **VADER Sentiment** — line-level sentiment scoring
- **NRC Emotion Lexicon (NRCLex)** — discrete emotion profiling
- **scikit-learn (TF-IDF, cosine similarity)** — content-based category matching
- **Streamlit** — interactive dashboard UI

## Tests

```bash
pip install pytest
pytest
```

The tests use small synthetic lyrics (no real songs) and cover the emotion scorer, query parsing, ranking, JSON schema and HTML escaping.

## Possible next steps

- Ground-truth evaluation: hand-label a small test set and measure actual classifier accuracy (and tune the retrieval weights and emotion priors against it)
- Swap the theme channel for sentence embeddings where a PyTorch wheel exists (`MoodIndex` keeps the channels separate, so this is a local change)
- Time-of-day-aware recommendations (map current time to a target mood category automatically)
- Multi-language support (current pipeline assumes English-language lyrics and lexicons)
