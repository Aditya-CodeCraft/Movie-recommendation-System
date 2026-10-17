"""Movie Mate: Streamlit front end for the content-based recommender in recommender.py."""
from pathlib import Path

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

        with st.container(border=True):
            for i, name in enumerate(names, start=1):
                st.markdown(f"#### &nbsp; {i}. {name}")

        st.balloons()
