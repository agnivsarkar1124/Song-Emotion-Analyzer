"""
Song Mood Explorer - build a curated playlist, see its emotional fingerprint,
and get recommendations by mood preference.

Run with: streamlit run app.py
(Note: live lyric fetching requires running locally - Genius blocks scraping
requests from shared cloud IP ranges, confirmed via diagnostic testing.)
"""

import json
import os
import time
import uuid

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import neural_ranker
import taste_ui
from mood_search import MoodIndex, playlist_to_json, song_key
from pipeline import (CALIBRATION, CATEGORY_TARGETS, SEED_SONGS, distance_similarity, format_duration,
                      get_lyrics, make_song_record)
from preferences import (ACTIONS, MAX_STORED_EVENTS, FeedbackEvent, ProfileStore, latest_change,
                         personalize, profile_history, song_states)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# --- Config -----------------------------------------------------------------

MAX_MOOD_FILE_BYTES = 20_000


# --- Core pipeline (Streamlit-free parts live in pipeline.py) -------------------

@st.cache_data(show_spinner=False)
def build_seed_lyrics(token):
    """{category: [(title, artist, lyrics), ...]} for the seed songs that could be fetched."""
    seeds = {}
    for category, songs in SEED_SONGS.items():
        fetched = []
        for title, artist in songs:
            try:
                fetched.append((title, artist, get_lyrics(title, artist, token, on_note=st.caption)))
                time.sleep(0.5)
            except ValueError:
                continue
        seeds[category] = fetched
    return seeds


@st.cache_data(show_spinner=False)
def build_reference_docs(token):
    return {
        category: " ".join(lyrics for _, _, lyrics in fetched)
        for category, fetched in build_seed_lyrics(token).items()
    }


@st.cache_data(show_spinner=False)
def build_seed_songs(token):
    """The seed songs as full library records, so they can be searched like the user's own."""
    return [
        make_song_record(title, artist, lyrics)
        for fetched in build_seed_lyrics(token).values()
        for title, artist, lyrics in fetched
    ]


def tfidf_similarity(lyrics, ref_text):
    if not ref_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([ref_text, lyrics])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def add_song_to_library(title, artist, token):
    return make_song_record(title, artist, get_lyrics(title, artist, token, on_note=st.caption))


def recommend_songs(library, category, k, reference_docs, cosine_weight=0.6):
    distance_weight = 1 - cosine_weight
    ref_text = reference_docs.get(category, "")

    def score_song(song):
        dist_sim = distance_similarity(song["mood"], song["variance"], category)
        sem_sim = tfidf_similarity(song["lyrics"], ref_text)
        combined = cosine_weight * sem_sim + distance_weight * dist_sim
        return {
            "title": song["title"],
            "artist": song["artist"],
            "category": song["category"],
            "mood": song["mood"],
            "variance": song["variance"],
            "match_score": round(combined, 3),
            "lyrics": song["lyrics"],  # carried through as grounding context for the RAG explanation step
            "is_exact_category": song["category"] == category,
        }

    scored = [score_song(song) for song in library]

    # Only recommend songs actually classified into the requested category -
    # ranking the whole library by raw similarity score (the old behavior)
    # could surface a song labeled "Sad" under a "Happy" search just because
    # its content score was high, which reads as a flat-out wrong result.
    exact_matches = sorted(
        [s for s in scored if s["is_exact_category"]],
        key=lambda x: x["match_score"], reverse=True
    )

    if len(exact_matches) >= k:
        return exact_matches[:k]

    # Not enough real matches in this category - fill the remainder with the
    # closest alternatives, but keep them clearly distinguishable so the UI
    # can label them honestly instead of silently mixing categories.
    fallback = sorted(
        [s for s in scored if not s["is_exact_category"]],
        key=lambda x: x["match_score"], reverse=True
    )
    return exact_matches + fallback[:k - len(exact_matches)]


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

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taste_profile.json")
PROFILE_STORE = ProfileStore(PROFILE_PATH)
MIX_KEY = "taste_mix"
EXAMPLE_MOODS = [
    "chill but slightly nostalgic for a rainy day",
    "sad but energetic",
    "angry gym playlist",
    "very romantic, not sad",
    "dreamy late-night drive",
]
ACTION_TOAST = {"like": "❤️ Liked", "dislike": "👎 Not for you:", "save": "⭐ Saved", "skip": "⏭️ Skipped",
                "unsave": "☆ Removed from saved:", "clear": "↺ Cleared your rating for"}

PAGE_CSS = """
<style>
.block-container { padding-top: 2rem; }
.hero h1 { font-size: 2.4rem; margin: 0; background: linear-gradient(90deg, #ffd166, #f15bb5 45%, #7aa2ff);
  -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 800; }
.hero p { margin: .2rem 0 1rem; opacity: .75; }
div[data-testid="stTabs"] button p { font-size: 1.02rem; font-weight: 600; }
</style>
"""


def record_feedback(action, key):
    """Turn one click into a stored FeedbackEvent (shared by the component and the fallback UI)."""
    if action not in ACTIONS or not key:
        return
    events = st.session_state.taste_events
    query = (st.session_state.get("mood_query") or {}).get("description", "")
    track = st.session_state.get("taste_tracks", {}).get(key)
    if track is not None:
        event = FeedbackEvent.create(action, track, query=query)
    else:  # e.g. "unsave" from the Saved tab for a song outside the current search
        previous = next((e for e in reversed(events) if e.key == key), None)
        if previous is None:
            return
        event = FeedbackEvent(action=action, key=key, title=previous.title, artist=previous.artist,
                              features=previous.features, query=query, ts=time.time(), id=uuid.uuid4().hex)
    events.append(event)
    del events[:-MAX_STORED_EVENTS]
    PROFILE_STORE.save(events)
    st.session_state.pending_toast = {"id": event.id, "action": action, "title": event.title}
    st.session_state.pop("model_check", None)   # an old cross-validation result is now stale


def on_mix_feedback():
    """Runs before the rerun that a click in the component triggers."""
    feedback = getattr(st.session_state.get(MIX_KEY), "feedback", None)
    if not isinstance(feedback, dict):
        return
    nonce = feedback.get("nonce")
    if nonce is not None and nonce == st.session_state.get("last_feedback_nonce"):
        return
    st.session_state.last_feedback_nonce = nonce
    record_feedback(feedback.get("action"), feedback.get("key"))


def use_example():
    st.session_state.mood_text = st.session_state.mood_example or st.session_state.get("mood_text", "")


def build_database(include_seeds):
    database = list(st.session_state.library)
    if include_seeds:
        known = {song_key(s["title"], s["artist"]) for s in database}
        database += [s for s in build_seed_songs(token) if song_key(s["title"], s["artist"]) not in known]
    return database


def get_index(database):
    """MoodIndex for this exact set of songs, rebuilt only when the set changes."""
    signature = tuple(song_key(s["title"], s["artist"]) for s in database)
    cached = st.session_state.get("mood_index_cache")
    if cached and cached[0] == signature:
        return cached[1]
    index = MoodIndex(database, CALIBRATION)
    st.session_state.mood_index_cache = (signature, index)
    return index


def toast_for(history):
    pending = st.session_state.get("pending_toast")
    if not pending:
        return None
    detail = None
    changes = latest_change(history)
    if changes and history and history[-1]["title"] == pending["title"]:
        detail = "Taste shift: " + " · ".join(("↑ " if c["delta"] > 0 else "↓ ") + c["label"] for c in changes)
    return {"id": pending["id"], "text": f"{ACTION_TOAST.get(pending['action'], pending['action'])} “{pending['title']}”",
            "detail": detail}


def fallback_mix(result):
    """Plain Streamlit version of the mix, for Streamlit builds without st.components.v2."""
    st.info("Your Streamlit version doesn't support the interactive view - upgrade Streamlit for the full experience.")
    for t in result["personal"]:
        cols = st.columns([5, 1, 1, 1, 1])
        change = t.get("rank_change") or 0
        arrow = f" (▲{change})" if change > 0 else f" (▼{-change})" if change < 0 else ""
        cols[0].markdown(f"**{t['rank']}. {t['title']}** — {t['artist']}{arrow}  \n{t['personal_percent']}% for you · {t['why']}")
        for col, (action, label) in zip(cols[1:], [("like", "❤️"), ("dislike", "👎"), ("save", "⭐"), ("skip", "⏭️")]):
            col.button(label, key=f"fb-{action}-{t['key']}", on_click=record_feedback, args=(action, t["key"]))
    insights = result["profile"]["insights"]
    if insights:
        st.markdown("**Your taste so far:** " + " · ".join(insights))


st.set_page_config(page_title="Song Mood Explorer", page_icon="🎵", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>🎵 Song Mood Explorer</h1>'
    "<p>Describe a mood, get a playlist from real lyrics, then like, nope, save or skip and watch it learn your taste.</p></div>",
    unsafe_allow_html=True,
)

if "library" not in st.session_state:
    st.session_state.library = []
if "taste_events" not in st.session_state:
    st.session_state.taste_events = PROFILE_STORE.load()

with st.sidebar:
    st.header("Setup")
    token = st.text_input("Genius API token", type="password", help="Get one free at genius.com/api-clients")
    anthropic_key = st.text_input(
        "Anthropic API key (optional)", type="password",
        help="Enables 'Explain this playlist' - a RAG-grounded explanation of the picks. Get one at console.anthropic.com"
    )
    st.markdown(f"**Songs in your library:** {len(st.session_state.library)}")
    if st.session_state.library and st.button("Clear library", use_container_width=True):
        st.session_state.library = []
        st.rerun()

    st.markdown("---")
    st.header("🧬 Your taste profile")
    events = st.session_state.taste_events
    states = song_states(events)
    rated = sum(1 for s in states.values() if s["liked"] or s["disliked"] or s["saved"] or s["skips"])
    st.markdown(f"**{rated}** songs rated · **{len(events)}** actions")
    info = neural_ranker.backend_info()
    st.caption(("🧠 Neural model: " if info["available"] else "📐 ") + info["label"])
    st.download_button(
        "⬇️ Download profile", json.dumps({"version": 1, "events": [e.__dict__ for e in events]}),
        file_name="taste_profile.json", mime="application/json", use_container_width=True, disabled=not events,
    )
    uploaded_profile = st.file_uploader("Load a profile", type=["json"], key="profile_upload")
    if uploaded_profile is not None and st.session_state.get("loaded_profile_id") != uploaded_profile.file_id:
        st.session_state.loaded_profile_id = uploaded_profile.file_id
        try:
            payload = json.loads(uploaded_profile.getvalue().decode("utf-8"))
            loaded = [FeedbackEvent.from_dict(d) for d in payload.get("events", [])
                      if isinstance(d, dict) and d.get("action") in ACTIONS and isinstance(d.get("features"), dict)]
            st.session_state.taste_events = loaded[-MAX_STORED_EVENTS:]
            PROFILE_STORE.save(st.session_state.taste_events)
            st.success(f"Loaded {len(loaded)} actions.")
            st.rerun()
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
            st.error("That file isn't a taste profile exported from this app.")
    if events and st.button("🗑️ Reset taste profile", use_container_width=True):
        st.session_state.taste_events = []
        st.session_state.pop("pending_toast", None)
        st.session_state.pop("model_check", None)
        PROFILE_STORE.clear()
        st.rerun()
    if events and info["available"] and st.button("🧪 Is the neural net helping?", use_container_width=True,
                                                  help="5-fold cross-validation on your own ratings: linear vs Wide & Deep"):
        st.session_state.model_check = "pending"
    st.caption("Feedback is saved to taste_profile.json next to app.py (no lyrics). "
               "Lyric fetching only works when running locally - Genius blocks cloud IPs.")

if not token:
    st.info("Enter a Genius API token in the sidebar to get started.")
    st.stop()

with st.spinner("Loading reference dataset (first run only, ~15 songs)..."):
    reference_docs = build_reference_docs(token)

tab_mix, tab_library, tab_presets = st.tabs(["🎧 Mood mix", "📚 Library", "🎚️ Preset moods"])

# --- Mood mix: describe -> retrieve -> personalise -> feedback loop ---
with tab_mix:
    desc_col, opt_col = st.columns([3, 1])
    with desc_col:
        if hasattr(st, "pills"):
            st.pills("Try one:", EXAMPLE_MOODS, key="mood_example", on_change=use_example)
        st.text_area("Describe a mood", key="mood_text", height=90, max_chars=500,
                     placeholder="e.g. chill but slightly nostalgic for a rainy day")
        mood_file = st.file_uploader("...or upload a .txt file with your description", type=["txt"])
    with opt_col:
        mood_k = st.slider("Tracks in playlist", min_value=1, max_value=15, value=5)
        include_seeds = st.checkbox(
            "Also search the 15 seed songs", value=True,
            help="The reference songs used to calibrate each mood category are searchable too, "
                 "so you get results even with an empty library.",
        )
        build_clicked = st.button("✨ Build my mix", type="primary", use_container_width=True)

    if build_clicked:
        description = (st.session_state.get("mood_text") or "").strip()
        if not description and mood_file is not None:
            raw = mood_file.getvalue()
            if len(raw) > MAX_MOOD_FILE_BYTES:
                st.warning(f"That file is larger than {MAX_MOOD_FILE_BYTES // 1000} KB - paste a shorter description.")
            else:
                description = raw.decode("utf-8", errors="ignore").strip()
        if not description:
            st.warning("Describe a mood first (type it above, pick an example, or upload a .txt file).")
        else:
            st.session_state.mood_query = {"description": description, "k": mood_k, "include_seeds": include_seeds}

    query = st.session_state.get("mood_query")
    if query:
        query["k"] = mood_k   # the slider re-ranks live
        with st.spinner("Loading songs..."):
            database = build_database(query["include_seeds"])
        mix = None
        if database:
            try:
                index = get_index(database)
                scored = index.score_all(query["description"])
                with st.spinner("Personalising..."):
                    mix = personalize(index, scored, st.session_state.taste_events, k=query["k"])
            except ValueError as e:
                st.warning(str(e))
        else:
            st.warning("The song database is empty - add songs in the Library tab or tick 'Also search the seed songs'.")

        if mix is not None:
            history = profile_history(st.session_state.taste_events, mix["population"])
            playlist = index.playlist(scored, mix["personal"])
            st.session_state.taste_tracks = {t["key"]: t for t in mix["base"] + mix["personal"] + mix["deck"]}
            songs_by_key = {s["key"]: s["src"] for s in index.songs}
            view = taste_ui.build_view(
                mix, playlist, history, latest_change(history),
                taste_ui.saved_items(mix["model"].states, songs_by_key),
                toast_for(history), version=len(st.session_state.taste_events), engine=neural_ranker.backend_info(),
            )
            if taste_ui.is_supported():
                taste_ui.render(view, key=MIX_KEY, on_feedback=on_mix_feedback)
            else:
                fallback_mix(mix)

            if st.session_state.get("model_check") == "pending":
                with st.spinner("Cross-validating linear vs neural on your ratings..."):
                    st.session_state.model_check = mix["model"].check() or "too few"
            check = st.session_state.get("model_check")
            if isinstance(check, dict):
                neural_acc = check["neural_accuracy"]
                st.info(f"🧪 On {check['n']} of your rated songs ({check['folds']}-fold cross-validation): "
                        f"linear model {check['linear_accuracy']:.0%} correct, "
                        f"Wide & Deep {neural_acc:.0%} correct." if neural_acc is not None else
                        f"🧪 Linear model {check['linear_accuracy']:.0%} correct on held-out ratings.")
            elif check == "too few":
                st.info("🧪 Rate at least 8 songs, including some you don't like, to compare the models.")

            act1, act2 = st.columns(2)
            with act1:
                st.download_button("⬇️ Download playlist JSON", playlist_to_json(playlist),
                                   file_name="mood_playlist.json", mime="application/json", use_container_width=True)
            with act2:
                explain_clicked = st.button("🤖 Explain this playlist", use_container_width=True)
            if explain_clicked:
                if not anthropic_key:
                    st.warning("Enter an Anthropic API key in the sidebar to use this feature.")
                else:
                    hits = [index.rag_hit(scored, next(j for j, s in enumerate(index.songs) if s["key"] == t["key"]),
                                          t["personal_percent"] / 100) for t in mix["personal"]]
                    with st.spinner("Generating a grounded explanation..."):
                        explanation, error = generate_rag_explanation(hits, query["description"], anthropic_key)
                    if error:
                        st.error(error)
                    else:
                        st.markdown("**Why these tracks fit your description:**")
                        st.write(explanation)
                        st.caption("Generated from the retrieved lyric excerpts and scores only - "
                                   "not general knowledge about these songs.")
    else:
        st.caption("Pick an example or describe a mood, then hit **Build my mix**. "
                   "Every like, nope, save and skip retrains your taste model and re-ranks the mix.")

# --- Library ---
with tab_library:
    st.subheader("Add a song")
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

    if st.session_state.library:
        st.subheader(f"Your library ({len(st.session_state.library)} songs)")
        st.dataframe(
            [
                {
                    "Title": s["title"],
                    "Artist": s["artist"],
                    "Length": format_duration(s.get("duration_seconds")),
                    "Words": s["word_count"],
                    "Mood": s["mood"],
                    "Variance": s["variance"],
                    "Category": s["category"],
                    "Top emotion": s["top_emotion"],
                    "Emotion conf.": s.get("emotion_confidence"),
                }
                for s in st.session_state.library
            ],
            use_container_width=True, hide_index=True,
        )
        with st.expander("🎧 Play preview clips"):
            for s in st.session_state.library:
                if s.get("preview_url"):
                    st.caption(f"**{s['title']}** - {s['artist']}")
                    st.audio(s["preview_url"])
                else:
                    st.caption(f"**{s['title']}** - {s['artist']} (no preview available)")
    else:
        st.info("Your library is empty. Add songs here - the more you add, the better your mixes get.")

# --- Preset moods (the original category recommender) ---
with tab_presets:
    if not st.session_state.library:
        st.info("Add a few songs in the Library tab, then get recommendations by preset mood category.")
    else:
        rec_col1, rec_col2 = st.columns([2, 1])
        with rec_col1:
            target_category = st.selectbox("Mood preference", options=list(CATEGORY_TARGETS.keys()))
        with rec_col2:
            top_k = st.number_input("How many songs", min_value=1, max_value=len(st.session_state.library),
                                    value=min(5, len(st.session_state.library)))

        if st.button("Recommend songs", use_container_width=True):
            with st.spinner("Scoring your library..."):
                recommendations = recommend_songs(st.session_state.library, target_category, int(top_k), reference_docs)
            st.session_state.last_recommendations = recommendations
            st.session_state.last_category = target_category

        recommendations = st.session_state.get("last_recommendations")
        if recommendations:
            category = st.session_state.last_category
            exact_count = sum(1 for r in recommendations if r["is_exact_category"])
            st.markdown(f"**Top {len(recommendations)} songs for '{category}':**")
            if exact_count < len(recommendations):
                st.caption(
                    f"Only {exact_count} song(s) in your library are actually classified as '{category}'. "
                    f"The rest below are the closest alternatives from other categories, marked accordingly."
                )
            st.dataframe(
                [
                    {
                        "Title": r["title"],
                        "Artist": r["artist"],
                        "Predicted category": r["category"],
                        "Match type": "Exact" if r["is_exact_category"] else "Closest alternative",
                        "Match score": r["match_score"],
                    }
                    for r in recommendations
                ],
                use_container_width=True, hide_index=True,
            )
            if st.button("🤖 Explain these picks", use_container_width=True):
                if not anthropic_key:
                    st.warning("Enter an Anthropic API key in the sidebar to use this feature.")
                else:
                    with st.spinner("Generating a grounded explanation..."):
                        explanation, error = generate_rag_explanation(recommendations, category, anthropic_key)
                    if error:
                        st.error(error)
                    else:
                        st.markdown("**Why these picks fit:**")
                        st.write(explanation)
                        st.caption("Generated from the actual retrieved lyric excerpts and scores above - "
                                   "not general knowledge about these songs.")
