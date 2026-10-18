"""Content-based movie recommendation: build a cosine-similarity matrix from text tags and look up the nearest titles.

The Streamlit app (apps.py) calls recommend() on the artifacts produced by the notebook. The tests exercise these
functions on a small synthetic catalogue, so they run without the TMDB dataset.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(tags: Sequence[str], max_features: int = 5000) -> np.ndarray:
    """Vectorise one tag string per movie as a bag of words and return the pairwise cosine-similarity matrix.

    This mirrors the notebook's CountVectorizer step so the tests can build a small catalogue. The notebook lower-cases
    and Porter-stems the tags before this step, which this function does not do, so it is not what produced
    similarity.pkl.

    A movie whose tags are empty, or made only of stop words, gets a zero vector. Its similarity to every movie,
    including itself, is 0, so it ranks after every movie that shares at least one word with the selected title and
    only shows up when k is larger than that; recommend() raises ValueError when asked about such a movie.
    """
    if len(tags) == 0:
        raise ValueError("tags must contain at least one movie")
    vectoriser = CountVectorizer(max_features=max_features, stop_words="english")
    try:
        vectors = vectoriser.fit_transform(tags)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):  # only relabel sklearn's "every document was empty or stop words" error
            raise
        raise ValueError("no usable words in any movie's tags") from exc
    return cosine_similarity(vectors)


def recommend(title: str, movies: pd.DataFrame, similarity: np.ndarray, k: int = 5) -> list[str]:
    """Return up to k titles most similar to `title`, most similar first, never including the movie's own row.

    `movies` must have a 'title' column whose row order matches the rows of `similarity`. The lookup uses the row
    position, not the index label, so a DataFrame whose index has gaps (for example after dropping rows during cleaning)
    still maps to the correct similarity row. If several rows share the title, the first one is used; the TMDB 5000 data
    has a handful of duplicate titles, and a different film that shares the selected title can still appear in the
    results. Unknown titles, and titles whose tags share no words with any other movie (so every score is 0), raise
    ValueError.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    matches = np.flatnonzero(movies["title"].to_numpy() == title)
    if len(matches) == 0:
        raise ValueError(f"unknown title: {title!r}")
    position = int(matches[0])

    scores = similarity[position]
    candidates = [(float(score), i) for i, score in enumerate(scores) if i != position]
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))  # highest score first; ties broken by row order
    if candidates and candidates[0][0] == 0.0:
        raise ValueError(f"no similar movies found for {title!r}: its tags share no words with any other movie")
    return [str(movies.iloc[i]["title"]) for _, i in candidates[:k]]


def load_artifacts(directory: str | Path = ".") -> tuple[pd.DataFrame, np.ndarray]:
    """Load movie_dict.pkl and similarity.pkl produced by the notebook.

    The DataFrame comes back with a fresh RangeIndex, so index labels equal row positions. Raises FileNotFoundError if
    either file is missing and ValueError if a file is empty or corrupt, the movie table has no 'title' column, the
    matrix is not square, or the two files disagree on the number of movies.
    """
    directory = Path(directory)
    try:
        with open(directory / "movie_dict.pkl", "rb") as handle:
            movies = pd.DataFrame(pickle.load(handle))
        with open(directory / "similarity.pkl", "rb") as handle:
            similarity = np.asarray(pickle.load(handle))
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError("a .pkl file is empty or corrupt; re-run prepare_data.ipynb to recreate it") from exc
    if "title" not in movies.columns:
        raise ValueError("movie_dict.pkl has no 'title' column")
    # The notebook drops rows without resetting the index, so the pickled labels have gaps; the dict keeps row order.
    movies = movies.reset_index(drop=True)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity.pkl must hold a square matrix")
    if len(movies) != similarity.shape[0]:
        raise ValueError("movie_dict.pkl and similarity.pkl have different numbers of movies")
    return movies, similarity


def google_search_url(title: str) -> str:
    """Return a Google search URL for "<title> movie", with the query URL-encoded.

    Spaces become "+" and characters such as "&", "<", apostrophes and accented letters are percent-encoded, so any
    title produces a single valid query string.
    """
    return f"https://www.google.com/search?q={quote_plus(title + ' movie')}"
