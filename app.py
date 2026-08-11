"""
Song Mood Explorer - build a curated playlist, see its emotional fingerprint,
and get recommendations by mood preference.

Run with: streamlit run app.py
(Note: live lyric fetching requires running locally - Genius blocks scraping
requests from shared cloud IP ranges, confirmed via diagnostic testing.)
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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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

SEED_SONGS = {
    "Happy": [("Happy", "Pharrell Williams"), ("Good as Hell", "Lizzo"), ("Walking on Sunshine", "Katrina and the Waves")],
    "Sad": [("Someone Like You", "Adele"), ("Skinny Love", "Bon Iver"), ("Someone You Loved", "Lewis Capaldi")],
    "Chill": [("Sunday Morning", "Maroon 5"), ("Banana Pancakes", "Jack Johnson"), ("Redbone", "Childish Gambino")],
    "Intense": [("Believer", "Imagine Dragons"), ("Radioactive", "Imagine Dragons"), ("Numb", "Linkin Park")],
    "Angry": [("Killing in the Name", "Rage Against the Machine"), ("Break Stuff", "Limp Bizkit"), ("Bodies", "Drowning Pool")],
}


# --- Core pipeline ------------------------------------------------------------

def fetch_with_retry(url, headers=None, params=None, retries=3, delay=2):
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
    data = response.json()

    hits = data.get("response", {}).get("hits", [])
    if not hits:
        raise ValueError(f"No Genius results found for '{song_title}' by '{artist}'")

    result = hits[0]["result"]
    matched_artist = result.get("artist_names", "")
    if artist.lower() not in matched_artist.lower():
        st.caption(f"Note: closest match was '{result['title']}' by '{matched_artist}' - verify this is the right song")

    song_url = result["url"]
    page = fetch_with_retry(song_url, headers={"User-Agent": headers["User-Agent"]})
    soup = BeautifulSoup(page.text, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})

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


def distance_similarity(mood, variance, category):
    target = CATEGORY_TARGETS[category]
    dist = math.sqrt((target["mood"] - mood) ** 2 + (target["variance"] - variance) ** 2)
    return 1 / (1 + dist)


def get_top_emotion(lyrics):
    if not NRCLEX_AVAILABLE:
        return None
    try:
        emotion = NRCLex(lyrics)
        scores = emotion.raw_emotion_scores
        if not scores:
            return None
        return max(scores, key=scores.get)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_reference_docs(token):
    reference_docs = {}
    for category, songs in SEED_SONGS.items():
        lyrics_list = []
        for title, artist in songs:
            try:
                lyrics_list.append(get_lyrics(title, artist, token))
                time.sleep(0.5)
            except ValueError:
                continue
        reference_docs[category] = " ".join(lyrics_list)
    return reference_docs


def tfidf_similarity(lyrics, ref_text):
    if not ref_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([ref_text, lyrics])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def get_track_info(title, artist):
    """
    Fetches duration and a 30-second preview clip URL via the iTunes Search
    API (free, no auth required). Returns (duration_seconds, preview_url),
    either of which may be None if unavailable.
    """
    params = {"term": f"{title} {artist}", "media": "music", "entity": "song", "limit": 1}
    try:
        response = requests.get("https://itunes.apple.com/search", params=params, timeout=10)
        results = response.json().get("results", [])
        if not results:
            return None, None
        track = results[0]
        millis = track.get("trackTimeMillis")
        duration = round(millis / 1000) if millis else None
        preview_url = track.get("previewUrl")
        return duration, preview_url
    except requests.exceptions.RequestException:
        return None, None


def format_duration(seconds):
    if seconds is None:
        return "-"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def add_song_to_library(title, artist, token):
    lyrics = get_lyrics(title, artist, token)
    mood, variance = get_song_mood(lyrics)
    category = nearest_category_by_mood(mood, variance)
    top_emotion = get_top_emotion(lyrics)
    word_count = len(lyrics.split())
    duration_seconds, preview_url = get_track_info(title, artist)

    return {
        "title": title,
        "artist": artist,
        "lyrics": lyrics,
        "mood": mood,
        "variance": variance,
        "category": category,
        "top_emotion": top_emotion or "-",
        "word_count": word_count,
        "duration_seconds": duration_seconds,
        "preview_url": preview_url,
    }


def recommend_songs(library, category, k, reference_docs, cosine_weight=0.6):
    distance_weight = 1 - cosine_weight
    ref_text = reference_docs.get(category, "")

    scored = []
    for song in library:
        dist_sim = distance_similarity(song["mood"], song["variance"], category)
        sem_sim = tfidf_similarity(song["lyrics"], ref_text)
        combined = cosine_weight * sem_sim + distance_weight * dist_sim
        scored.append({
            "title": song["title"],
            "artist": song["artist"],
            "category": song["category"],
            "mood": song["mood"],
            "variance": song["variance"],
            "match_score": round(combined, 3),
            "lyrics": song["lyrics"],  # carried through as grounding context for the RAG explanation step
        })

    return sorted(scored, key=lambda x: x["match_score"], reverse=True)[:k]


# --- RAG: generate a grounded explanation for the retrieved recommendations ---
#
# This is a standard Retrieve -> Augment -> Generate pipeline:
#   Retrieve:  recommend_songs() above already does this - TF-IDF + mood-distance
#              scoring against the user's library, no LLM involved.
#   Augment:   the retrieved songs' actual lyrics are inserted into the prompt
#              as grounding context, so the model can only reason about real,
#              retrieved content - not invent facts about songs it wasn't given.
#   Generate:  an LLM turns that grounded context into a natural-language
#              explanation of why these specific songs fit the requested mood.
def build_rag_context(recommendations, max_words_per_song=60):
    """Truncates each recommended song's lyrics to a short excerpt, keeping
    the prompt small and avoiding reproducing full lyrics verbatim anywhere,
    including inside the LLM's context window."""
    context_blocks = []
    for i, song in enumerate(recommendations, start=1):
        words = song["lyrics"].split()
        excerpt = " ".join(words[:max_words_per_song])
        context_blocks.append(
            f"[{i}] \"{song['title']}\" by {song['artist']} "
            f"(mood={song['mood']}, variance={song['variance']}, match_score={song['match_score']})\n"
            f"Lyric excerpt: {excerpt}"
        )
    return "\n\n".join(context_blocks)


def generate_rag_explanation(recommendations, category, api_key):
    """
    Calls Claude to explain, in plain language, why the retrieved songs fit
    the requested mood - grounded strictly in the retrieved lyric excerpts
    and computed scores, not general knowledge about the songs.
    Returns (explanation_text, error_message) - exactly one will be None.
    """
    if not ANTHROPIC_AVAILABLE:
        return None, "The anthropic package isn't installed (pip install anthropic)."
    if not api_key:
        return None, "No Anthropic API key provided."

    context = build_rag_context(recommendations)

    system_prompt = (
        "You explain why specific songs were recommended for a given mood, "
        "using ONLY the lyric excerpts and scores provided below - do not use "
        "outside knowledge about these songs, and do not quote more than a few "
        "words of any lyric verbatim. Paraphrase themes instead of quoting. "
        "Be concise: 2-3 sentences per song, plain language, no headers."
    )
    user_prompt = (
        f"Mood requested: {category}\n\n"
        f"Retrieved songs (already ranked by a similarity/distance algorithm, "
        f"not by you):\n\n{context}\n\n"
        f"Explain briefly why each song's lyrical content and mood/variance "
        f"scores make it a fit for '{category}'."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        explanation = "".join(block.text for block in response.content if hasattr(block, "text"))
        return explanation, None
    except Exception as e:
        return None, f"Couldn't generate an explanation: {e}"


# --- Streamlit UI ---------------------------------------------------------------

st.set_page_config(page_title="Song Mood Explorer", page_icon="🎵", layout="centered")

st.title("🎵 Song Mood Explorer")
st.caption(
    "Build a curated playlist, see each song's emotional fingerprint, and get "
    "recommendations by mood - powered by VADER sentiment, the NRC emotion "
    "lexicon, and TF-IDF content similarity."
)

if "library" not in st.session_state:
    st.session_state.library = []

with st.sidebar:
    st.header("Setup")
    token = st.text_input("Genius API token", type="password", help="Get one free at genius.com/api-clients")
    anthropic_key = st.text_input(
        "Anthropic API key (optional)", type="password",
        help="Enables 'Explain these picks' - a RAG-grounded explanation of your recommendations. Get one at console.anthropic.com"
    )
    st.markdown("---")
    st.markdown(f"**Songs in your library:** {len(st.session_state.library)}")
    if st.session_state.library and st.button("Clear library", use_container_width=True):
        st.session_state.library = []
        st.rerun()
    st.markdown("---")
    st.caption(
        "Note: live lyric fetching only works when running locally. "
        "Genius blocks scraping requests from shared cloud hosting IPs."
    )

if not token:
    st.info("Enter a Genius API token in the sidebar to get started.")
    st.stop()

with st.spinner("Loading reference dataset (first run only, ~15 songs)..."):
    reference_docs = build_reference_docs(token)

# --- Add songs ---
st.subheader("Add a song to your library")
col1, col2, col3 = st.columns([3, 3, 1])
with col1:
    song_title = st.text_input("Song title", placeholder="e.g. Bad Habits", label_visibility="collapsed")
with col2:
    artist = st.text_input("Artist", placeholder="e.g. Ed Sheeran", label_visibility="collapsed")
with col3:
    add_clicked = st.button("Add", type="primary", use_container_width=True)

if add_clicked:
    if not song_title or not artist:
        st.warning("Enter both a song title and an artist.")
    elif any(s["title"].lower() == song_title.lower() and s["artist"].lower() == artist.lower() for s in st.session_state.library):
        st.warning(f"'{song_title}' by {artist} is already in your library.")
    else:
        try:
            with st.spinner(f"Fetching '{song_title}' by {artist}..."):
                new_song = add_song_to_library(song_title, artist, token)
            st.session_state.library.append(new_song)
            st.success(f"Added '{song_title}' by {artist} - classified as **{new_song['category']}**")
        except ValueError as e:
            st.error(str(e))
        except requests.exceptions.RequestException as e:
            st.error(f"Network error reaching Genius: {e}")

# --- Library table ---
if st.session_state.library:
    st.subheader(f"Your library ({len(st.session_state.library)} songs)")
    table_data = [
        {
            "Title": s["title"],
            "Artist": s["artist"],
            "Length": format_duration(s.get("duration_seconds")),
            "Words": s["word_count"],
            "Mood": s["mood"],
            "Variance": s["variance"],
            "Category": s["category"],
            "Top emotion": s["top_emotion"],
        }
        for s in st.session_state.library
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    with st.expander("🎧 Play preview clips"):
        for s in st.session_state.library:
            if s.get("preview_url"):
                st.caption(f"**{s['title']}** - {s['artist']}")
                st.audio(s["preview_url"])
            else:
                st.caption(f"**{s['title']}** - {s['artist']} (no preview available)")

    # --- Recommendations ---
    st.subheader("Get recommendations by mood")
    rec_col1, rec_col2 = st.columns([2, 1])
    with rec_col1:
        target_category = st.selectbox("Mood preference", options=list(CATEGORY_TARGETS.keys()))
    with rec_col2:
        top_k = st.number_input("How many songs", min_value=1, max_value=len(st.session_state.library), value=min(5, len(st.session_state.library)))

    if st.button("Recommend songs", use_container_width=True):
        with st.spinner("Scoring your library..."):
            recommendations = recommend_songs(st.session_state.library, target_category, int(top_k), reference_docs)

        st.session_state.last_recommendations = recommendations
        st.session_state.last_category = target_category

        st.markdown(f"**Top {len(recommendations)} songs for '{target_category}':**")
        st.dataframe(
            [
                {
                    "Title": r["title"],
                    "Artist": r["artist"],
                    "Predicted category": r["category"],
                    "Match score": r["match_score"],
                }
                for r in recommendations
            ],
            use_container_width=True,
            hide_index=True,
        )

    # --- RAG explanation: generation step, grounded in the retrieval above ---
    if st.session_state.get("last_recommendations"):
        if st.button("🤖 Explain these picks", use_container_width=True):
            if not anthropic_key:
                st.warning("Enter an Anthropic API key in the sidebar to use this feature.")
            else:
                with st.spinner("Generating a grounded explanation..."):
                    explanation, error = generate_rag_explanation(
                        st.session_state.last_recommendations,
                        st.session_state.last_category,
                        anthropic_key,
                    )
                if error:
                    st.error(error)
                else:
                    st.markdown("**Why these picks fit:**")
                    st.write(explanation)
                    st.caption(
                        "Generated from the actual retrieved lyric excerpts and scores above - "
                        "not general knowledge about these songs."
                    )
else:
    st.info("Add a few songs above to build your library, then get mood-based recommendations.")
