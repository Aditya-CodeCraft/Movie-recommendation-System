import streamlit as st
import pandas as pd
import pickle
import time

# --- Page Configuration ---
# Sets a title and icon for the browser tab, and centers the layout.
st.set_page_config(
    page_title="Movie Mate",
    page_icon="🎬",
    layout="centered"
)

# --- Sidebar ---
# Adds a sidebar with information about the app.
with st.sidebar:
    st.title("About Movie Mate")
    st.info(
        "This is a content-based movie recommender system. "
        "Select a movie you like, and the app will suggest "
        "5 other movies with similar content and themes."
    )
    st.success("Built with Streamlit by a budding developer!")


# --- Load Data ---
# This part remains the same.
@st.cache_data  # Caches the data to speed up app loading
def load_data():
    try:
        movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except FileNotFoundError:
        return None, None


movies, similarity = load_data()

if movies is None:
    st.error("Data files not found! Make sure 'movie_dict.pkl' and 'similarity.pkl' are in the folder.")
    st.stop()


# --- Recommendation Logic ---
# This function also remains the same.
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]
    return recommended_movies


# --- Main App Interface ---
st.title('🎬 Movie Mate')
st.markdown("### Find Your Next Favorite Movie!")

# Create a container for the selection box for better organization.
input_container = st.container()
with input_container:
    selected_movie_name = st.selectbox(
        'Start by selecting a movie you like:',
        movies['title'].values,
        index=None,  # Defaults to nothing selected
        placeholder="Type or select a movie..."
    )

# Only show the button if a movie has been selected.
if selected_movie_name:
    if st.button(f'🍿 Recommend Movies Like "{selected_movie_name}"', use_container_width=True):
        with st.spinner('Analyzing similarities and finding matches...'):
            time.sleep(1)  # A small delay for a better user experience
            names = recommend(selected_movie_name)

        # --- Improved Display for Recommendations ---
        st.subheader("Here are your top recommendations:")

        # Using a container with a border to display results cleanly.
        with st.container(border=True):
            for i, name in enumerate(names):
                st.markdown(f"#### &nbsp; {i + 1}. {name}")  # Indent for style

        # A fun animation to celebrate!
        st.balloons()