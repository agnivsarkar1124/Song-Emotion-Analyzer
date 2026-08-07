"""
Song Mood Explorer - a standalone interactive dashboard.

Enter a song + artist, and it fetches the lyrics, scores mood/variance
(VADER), profiles discrete emotions (NRCLex), and classifies it into a
mood category using a small built-in seed dataset as reference data.

Run with: streamlit run app.py
"""

import math
import statistics
import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False


# --- Config -----------------------------------------------------------------

CATEGORY_TARGETS = {
    "Happy":   {"mood": 0.33, "variance": 0.15},
    "Sad":     {"mood": -0.07, "variance": 0.15},
    "Chill":   {"mood": 0.17, "variance": 0.10},
    "Intense": {"mood": 0.17, "variance": 0.23},
    "Angry":   {"mood": -0.07, "variance": 0.23},
}

# A small seed dataset (3 songs per category) used to build the reference
# documents for TF-IDF comparison. Kept small so a fresh clone of this repo
# only needs ~15 Genius fetches on first run, not hundreds.
SEED_SONGS = {
    "Happy": [("Happy", "Pharrell Williams"), ("Good as Hell", "Lizzo"), ("Walking on Sunshine", "Katrina and the Waves")],
    "Sad": [("Someone Like You", "Adele"), ("Skinny Love", "Bon Iver"), ("Someone You Loved", "Lewis Capaldi")],
    "Chill": [("Sunday Morning", "Maroon 5"), ("Banana Pancakes", "Jack Johnson"), ("Redbone", "Childish Gambino")],
    "Intense": ["Believer", "Radioactive", "Numb"],  # placeholder, fixed below
    "Angry": [("Killing in the Name", "Rage Against the Machine"), ("Break Stuff", "Limp Bizkit"), ("Bodies", "Drowning Pool")],
}
# fix the malformed Intense entry above
SEED_SONGS["Intense"] = [("Believer", "Imagine Dragons"), ("Radioactive", "Imagine Dragons"), ("Numb", "Linkin Park")]


# --- Core pipeline (ported from the main project) ----------------------------

def fetch_with_retry(url, headers=None, params=None, retries=3, delay=2):
    """
    Wraps requests.get with retries. Without this, a transient network
    hiccup (connection reset, brief timeout) fails outright instead of
    recovering - which showed up as songs silently coming back with
    empty lyrics rather than a clear network error.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return requests.get(url, headers=headers, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_error


def get_lyrics(song_title, artist, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    }
    params = {"q": f"{song_title} {artist}"}
    response = fetch_with_retry("https://api.genius.com/search", headers=headers, params=params)
    print(f"[DEBUG] Search status: {response.status_code}")
    data = response.json()

    hits = data.get("response", {}).get("hits", [])
    print(f"[DEBUG] Hits found: {len(hits)}")
    if not hits:
        raise ValueError(f"No Genius results found for '{song_title}' by '{artist}'")

    result = hits[0]["result"]
    song_url = result["url"]
    print(f"[DEBUG] Matched URL: {song_url}")

    page = fetch_with_retry(song_url, headers={"User-Agent": headers["User-Agent"]})
    print(f"[DEBUG] Page fetch status: {page.status_code}, length: {len(page.text)}")

    soup = BeautifulSoup(page.text, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    print(f"[DEBUG] Lyrics containers found: {len(containers)}")

    lyrics_parts = []
    for container in containers:
        for junk in container.find_all(attrs={"data-exclude-from-selection": "true"}):
            junk.decompose()
        lyrics_parts.append(container.get_text(separator="\n"))

    lyrics = "\n".join(lyrics_parts)
    if not lyrics.strip():
        raise ValueError(f"Lyrics came back empty for '{song_title}' by '{artist}' - check the title/artist spelling")
    return lyrics


_analyzer = SentimentIntensityAnalyzer()

def get_song_mood(lyrics):
    lines = [line for line in lyrics.split("\n") if line.strip()]
    line_scores = [_analyzer.polarity_scores(line)["compound"] for line in lines]
    compounds = [s for s in line_scores if abs(s) > 0.05]

    avg_mood = sum(compounds) / len(compounds) if compounds else 0
    mood_variance = statistics.variance(compounds) if len(compounds) > 1 else 0
    return round(avg_mood, 2), round(mood_variance, 2)


def nearest_category_by_mood(mood, variance):
    def distance(cat):
        target = CATEGORY_TARGETS[cat]
        return math.sqrt((target["mood"] - mood) ** 2 + (target["variance"] - variance) ** 2)
    return min(CATEGORY_TARGETS, key=distance)


def get_emotion_profile(lyrics):
    if not NRCLEX_AVAILABLE:
        return None
    try:
        emotion = NRCLex(lyrics)
        scores = emotion.raw_emotion_scores
        total = sum(scores.values()) if scores else 1
        return {k: round(v / total, 3) for k, v in scores.items()}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_reference_docs(token):
    """Fetches the small seed dataset once and caches it for the session."""
    reference_docs = {}
    for category, songs in SEED_SONGS.items():
        lyrics_list = []
        for title, artist in songs:
            try:
                lyrics_list.append(get_lyrics(title, artist, token))
                time.sleep(0.5)
            except ValueError:
                continue  # skip a seed song if Genius mismatches or fails
        reference_docs[category] = " ".join(lyrics_list)
    return reference_docs


def category_similarity_scores(lyrics, reference_docs):
    scores = {}
    for category, ref_text in reference_docs.items():
        if not ref_text.strip():
            scores[category] = 0.0
            continue
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([ref_text, lyrics])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        scores[category] = round(float(sim), 3)
    return scores


# --- Streamlit UI -------------------------------------------------------------

st.set_page_config(page_title="Song Mood Explorer", page_icon="🎵", layout="centered")

st.title("🎵 Song Mood Explorer")
st.caption(
    "Enter a song and artist to see its mood, emotional profile, and predicted "
    "category - built on VADER sentiment, NRC emotion lexicon, and TF-IDF "
    "similarity against a small reference set."
)

with st.sidebar:
    st.header("Setup")
    token = st.text_input("Genius API token", type="password", help="Get one free at genius.com/api-clients")
    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Fetches lyrics via the Genius API\n"
        "2. Scores mood + emotional volatility (VADER, line-by-line)\n"
        "3. Profiles discrete emotions (NRC Emotion Lexicon)\n"
        "4. Classifies via distance to mood targets *and* content "
        "similarity (TF-IDF) to a small reference set per category"
    )

if not token:
    st.info("Enter a Genius API token in the sidebar to get started.")
    st.stop()

with st.spinner("Loading reference dataset (first run only, ~15 songs)..."):
    reference_docs = build_reference_docs(token)

col1, col2 = st.columns(2)
with col1:
    song_title = st.text_input("Song title", placeholder="e.g. Bad Habits")
with col2:
    artist = st.text_input("Artist", placeholder="e.g. Ed Sheeran")

if st.button("Analyze", type="primary", use_container_width=True):
    if not song_title or not artist:
        st.warning("Enter both a song title and an artist.")
        st.stop()

    try:
        with st.spinner(f"Fetching '{song_title}' by {artist}..."):
            lyrics = get_lyrics(song_title, artist, token)
            mood, variance = get_song_mood(lyrics)
            category = nearest_category_by_mood(mood, variance)
            sim_scores = category_similarity_scores(lyrics, reference_docs)
            emotion_profile = get_emotion_profile(lyrics)

        st.success(f"Analyzed **{song_title}** by **{artist}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mood", mood, help="-1 (negative) to +1 (positive), averaged across lines")
        m2.metric("Variance", variance, help="How much emotional swing across the song's lines")
        m3.metric("Predicted category", category)

        st.subheader("Content similarity by category")
        st.caption("How similar this song's lyrics are to each category's reference songs (TF-IDF cosine similarity)")
        st.bar_chart(sim_scores)

        if emotion_profile:
            st.subheader("Emotion profile (NRC Lexicon)")
            st.caption("Proportion of emotion-tagged words in the lyrics, by category")
            st.bar_chart(emotion_profile)
        elif NRCLEX_AVAILABLE:
            st.info("NRCLex ran but returned no scores for this song (short or ambiguous lyrics).")
        else:
            st.info("NRCLex isn't installed - run `pip install nrclex` to enable emotion profiling.")

        with st.expander("View raw lyrics"):
            st.text(lyrics)

    except ValueError as e:
        st.error(str(e))
    except requests.exceptions.RequestException as e:
        st.error(f"Network error reaching Genius: {e}")
