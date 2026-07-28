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

## Try it

**[Live demo →](#)** *(add your deployed Streamlit link here)*

Or run locally:
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

## Possible next steps

- Ground-truth evaluation: hand-label a small test set and measure actual classifier accuracy
- Time-of-day-aware recommendations (map current time to a target mood category automatically)
- Multi-language support (current pipeline assumes English-language lyrics and lexicons)
