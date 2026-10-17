"""Edge-case tests for recommender.py on a tiny synthetic catalogue (no TMDB download needed)."""
import pickle

import numpy as np
import pandas as pd
import pytest

from recommender import build_similarity, load_artifacts, recommend

# Two clear clusters (space films, romances) plus one movie with no usable metadata.
CATALOGUE = {
    "Space Battle": "space war fleet galaxy rebellion starship",
    "Galaxy War": "space war galaxy empire starship battle",
    "Quiet Romance": "love letters village summer romance",
    "Second Romance": "love village summer romance wedding",
    "Empty Metadata": "",
}


@pytest.fixture
def catalogue():
    movies = pd.DataFrame({"title": list(CATALOGUE)})
    similarity = build_similarity(list(CATALOGUE.values()))
    return movies, similarity


def test_similarity_matrix_is_square_and_matches_catalogue(catalogue):
    movies, similarity = catalogue
    assert similarity.shape == (len(movies), len(movies))


def test_recommendations_never_include_the_movie_itself(catalogue):
    movies, similarity = catalogue
    for title in ("Space Battle", "Galaxy War", "Quiet Romance", "Second Romance"):
        assert title not in recommend(title, movies, similarity, k=4)


def test_most_similar_movie_is_ranked_first(catalogue):
    movies, similarity = catalogue
    assert recommend("Space Battle", movies, similarity, k=1) == ["Galaxy War"]
    assert recommend("Quiet Romance", movies, similarity, k=1) == ["Second Romance"]


def test_results_are_ordered_by_descending_similarity(catalogue):
    movies, similarity = catalogue
    position = list(movies["title"]).index("Space Battle")
    titles = recommend("Space Battle", movies, similarity, k=4)
    scores = [similarity[position][list(movies["title"]).index(t)] for t in titles]
    assert scores == sorted(scores, reverse=True)


def test_unknown_title_raises_a_clear_error(catalogue):
    movies, similarity = catalogue
    with pytest.raises(ValueError, match="unknown title"):
        recommend("Not In Catalogue", movies, similarity)


def test_k_larger_than_catalogue_returns_everything_else_without_padding(catalogue):
    movies, similarity = catalogue
    result = recommend("Space Battle", movies, similarity, k=50)
    assert len(result) == len(movies) - 1
    assert len(set(result)) == len(result)


def test_k_must_be_positive(catalogue):
    movies, similarity = catalogue
    with pytest.raises(ValueError, match="k must be at least 1"):
        recommend("Space Battle", movies, similarity, k=0)


def test_movie_with_empty_metadata_has_no_nan_scores_and_is_never_recommended_first(catalogue):
    movies, similarity = catalogue
    assert not np.isnan(similarity).any()
    # Empty tags give a zero vector, so its similarity row (including the diagonal) is all zeros.
    assert not similarity[list(CATALOGUE).index("Empty Metadata")].any()
    for title in ("Space Battle", "Galaxy War", "Quiet Romance", "Second Romance"):
        # A zero vector scores 0 against everything, so it can only appear after every movie with real overlap.
        assert recommend(title, movies, similarity, k=1) != ["Empty Metadata"]
    # Asking for recommendations for the empty movie is reported clearly instead of returning arbitrary titles.
    with pytest.raises(ValueError, match="share no words"):
        recommend("Empty Metadata", movies, similarity, k=2)


def test_duplicate_titles_resolve_to_the_first_matching_row():
    titles = ["Space Battle", "Galaxy War", "Space Battle", "Quiet Romance"]
    tags = [
        "space war fleet galaxy rebellion starship",
        "space war galaxy empire starship battle",
        "love letters village summer romance",  # a second, unrelated film with the same title
        "love village summer romance wedding",
    ]
    movies = pd.DataFrame({"title": titles})
    similarity = build_similarity(tags)
    # The first "Space Battle" row is the space film, so its nearest neighbour is "Galaxy War".
    assert recommend("Space Battle", movies, similarity, k=1) == ["Galaxy War"]


def test_non_default_index_still_maps_to_the_correct_similarity_row(catalogue):
    movies, similarity = catalogue
    shifted = movies.copy()
    shifted.index = range(100, 100 + len(shifted))  # index labels no longer equal row positions
    assert recommend("Space Battle", shifted, similarity, k=4) == recommend("Space Battle", movies, similarity, k=4)


def test_all_empty_tags_raise_a_clear_error():
    with pytest.raises(ValueError, match="no usable words"):
        build_similarity(["", "", ""])


def test_empty_catalogue_is_rejected():
    with pytest.raises(ValueError, match="at least one movie"):
        build_similarity([])


def test_load_artifacts_round_trips_notebook_pickles_with_index_gaps(tmp_path):
    # The notebook drops rows without reset_index, so the pickled dict carries index labels with gaps.
    new_df = pd.DataFrame(
        {"movie_id": [1, 2, 3], "title": ["A", "B", "C"], "tags": ["alpha beta", "alpha gamma", "gamma delta"]},
        index=[0, 2, 3],
    )
    (tmp_path / "movie_dict.pkl").write_bytes(pickle.dumps(new_df.to_dict()))
    (tmp_path / "similarity.pkl").write_bytes(pickle.dumps(build_similarity(new_df["tags"])))
    movies, similarity = load_artifacts(tmp_path)
    assert list(movies["title"]) == ["A", "B", "C"]
    assert list(movies.index) == [0, 1, 2]  # labels equal positions again
    assert similarity.shape == (3, 3)
    assert recommend("C", movies, similarity, k=1) == ["B"]  # row 2; a label-based lookup would have asked for row 3


def test_load_artifacts_rejects_mismatched_sizes(tmp_path):
    (tmp_path / "movie_dict.pkl").write_bytes(pickle.dumps({"title": {0: "A", 1: "B"}}))
    (tmp_path / "similarity.pkl").write_bytes(pickle.dumps(np.eye(3)))
    with pytest.raises(ValueError, match="different numbers of movies"):
        load_artifacts(tmp_path)


def test_load_artifacts_reports_an_empty_or_corrupt_pickle(tmp_path):
    (tmp_path / "movie_dict.pkl").write_bytes(pickle.dumps({"title": {0: "A"}}))
    (tmp_path / "similarity.pkl").write_bytes(b"")
    with pytest.raises(ValueError, match="empty or corrupt"):
        load_artifacts(tmp_path)
