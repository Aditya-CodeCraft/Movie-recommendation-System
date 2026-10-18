"""Movie Mate: Streamlit front end for the content-based recommender in recommender.py."""
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st

from recommender import load_artifacts, recommend

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Mate",
    page_icon="🎬",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.title("About Movie Mate")
    st.info(
        "This is a content-based movie recommender system. "
        "Select a movie you like, and the app will suggest "
        "5 other movies with similar content and themes."
    )
    st.caption("Built by Aditya Gupta. Source: https://github.com/Aditya-CodeCraft/Movie-recommendation-System")


# --- Load Data ---
@st.cache_resource  # one shared read-only copy per server process; cache_data would re-pickle the 180 MB matrix on every rerun
def load_data():
    return load_artifacts(Path(__file__).resolve().parent)  # the .pkl files live next to this script, whatever the CWD


try:
    movies, similarity = load_data()
except FileNotFoundError:
    st.error("Data files not found! Run prepare_data.ipynb to create 'movie_dict.pkl' and 'similarity.pkl' next to apps.py.")
    st.stop()
except ValueError as exc:
    st.error(f"Data files are inconsistent: {exc}")
    st.stop()


# --- Main App Interface ---
st.title("🎬 Movie Mate")
st.markdown("### Find Your Next Favorite Movie!")

selected_movie_name = st.selectbox(
    "Start by selecting a movie you like:",
    movies["title"].tolist(),
    index=None,  # Defaults to nothing selected
    placeholder="Type or select a movie...",
)

# Only show the button once a movie has been selected.
if selected_movie_name:
    if st.button(f'🍿 Recommend Movies Like "{selected_movie_name}"', width="stretch"):
        with st.spinner("Analyzing similarities and finding matches..."):
            try:
                names = recommend(selected_movie_name, movies, similarity, k=5)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.subheader("Here are your top recommendations:")

        # Streamlit's own heading anchor only jumps within the page, so draw the hover link ourselves: it opens a
        # Google search for the movie in a new tab.
        st.markdown(
            """<style>
            .mm-rec { font-size: 1.5rem; font-weight: 600; line-height: 1.2; padding: 0.75rem 0 1rem; }
            .mm-rec a { opacity: 0; margin-left: 0.5rem; color: inherit; vertical-align: middle; transition: opacity 0.15s; }
            .mm-rec:hover a, .mm-rec a:focus { opacity: 1; }
            </style>""",
            unsafe_allow_html=True,
        )
        link_icon = (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07'
            'l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>'
        )
        with st.container(border=True):
            for i, name in enumerate(names, start=1):
                google_url = f"https://www.google.com/search?q={quote_plus(name + ' movie')}"
                st.markdown(
                    f'<div class="mm-rec">&nbsp; {i}. {escape(name)}'
                    f'<a href="{escape(google_url)}" target="_blank" rel="noopener" '
                    f'title="Search Google for {escape(name)}">{link_icon}</a></div>',
                    unsafe_allow_html=True,
                )

        st.balloons()
