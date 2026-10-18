# Movie Mate: content-based movie recommender

[![tests](https://github.com/Aditya-CodeCraft/Movie-recommendation-System/actions/workflows/tests.yml/badge.svg)](https://github.com/Aditya-CodeCraft/Movie-recommendation-System/actions/workflows/tests.yml)

Pick a movie you like and Movie Mate suggests five similar titles, using only the movies' own metadata: overview, genres, keywords, top-billed cast and director from the TMDB 5000 dataset. Streamlit front end, scikit-learn underneath.

## How it works

1. **Prepare the data** (`prepare_data.ipynb`). Merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on title, drop the three movies with no overview, parse the JSON-like `genres`, `keywords`, `cast` (top three) and `crew` (director) columns, collapse multi-word names into single tokens (`Sam Worthington` becomes `samworthington`, so first names do not match across people), join them with the overview into one `tags` string per movie, lower-case it and stem every word with NLTK's PorterStemmer (which is why `nltk` is in `requirements.txt`).
2. **Vectorise.** `CountVectorizer(max_features=5000, stop_words="english")` turns each `tags` string into a bag-of-words vector.
3. **Measure similarity.** Cosine similarity between every pair of movies gives a square matrix over the roughly 4,800 movies, saved with pickle as `similarity.pkl` next to the movie list `movie_dict.pkl`.
4. **Recommend** (`recommender.py`). Look up the selected title's row, sort the other movies by similarity and return the top five. The lookup is by row position: an earlier version used the DataFrame index label as the row number, which pointed at the wrong similarity row for every title after the three movies dropped during cleaning (and raised `IndexError` for the last three). The Streamlit app (`apps.py`) wraps this in a searchable select box and a results panel.
5. **Look a movie up.** Each recommended title is a link that opens a Google search for that movie in a new tab, and a link icon appears when you hover over it. `google_search_url()` in `recommender.py` builds the URL; titles are HTML-escaped on the page and URL-encoded in the query, so names with `&`, `<`, apostrophes or accents display and link correctly.

## Project structure

| Path | Purpose |
|---|---|
| `apps.py` | Streamlit user interface |
| `recommender.py` | `recommend()`, `load_artifacts()` and `google_search_url()`, used by the app, plus `build_similarity()`, a small re-implementation of the notebook's CountVectorizer step (without stemming) so the tests can build a catalogue without the dataset |
| `tests/test_recommender.py` | pytest suite covering the edge cases listed below |
| `prepare_data.ipynb` | Data preparation and artifact generation |
| `requirements.txt`, `requirements-dev.txt` | Runtime and test dependencies |

## Setup

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data) from Kaggle and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project folder.
2. Run `prepare_data.ipynb` to create `movie_dict.pkl` and `similarity.pkl`. They are git-ignored because the similarity matrix is roughly 180 MB.
3. Install and run:

```bash
pip install -r requirements.txt
streamlit run apps.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests build a five-movie synthetic catalogue, so they run in a couple of seconds without the dataset (most of that is importing scikit-learn). They check that:

- the selected movie's own row is never in its recommendations;
- the most similar movie is ranked first, and results are in descending similarity order;
- an unknown title raises a `ValueError` with a clear message (which `apps.py` shows as an error box rather than a traceback);
- `k` larger than the catalogue returns everything else, with no padding or duplicates;
- a movie with empty metadata gets an all-zero similarity row, produces no NaN scores, is never the first recommendation, and asking for recommendations for it raises a clear error instead of returning arbitrary titles;
- duplicate titles resolve to the first matching row;
- a DataFrame whose index labels differ from the row positions still maps to the correct similarity row (the lookup is positional, so dropping rows during cleaning cannot shift the results);
- `load_artifacts()` reads the notebook's pickles (a dict whose index labels have gaps) back into a table whose positions line up with the matrix, and rejects mismatched sizes and empty or corrupt files;
- an empty list of movies, or a catalogue in which every movie's tags are empty, is rejected with a clear message.
- `google_search_url()` encodes spaces, `&`, `<`, apostrophes and accented letters correctly.

## Known limitations

- Recommendations are only as good as the tags. Movies with short overviews or missing cast data get weaker matches, and a movie whose tags share no words with any other movie gets no recommendations at all (the app says so instead of listing arbitrary titles).
- Three TMDB titles belong to two different films each. Because the data is merged on `title`, each of them appears four times in the catalogue, twice paired with the other film's cast and director; the first matching row wins when you search by title, and the other film with the same name can show up among the results. Merging on the TMDB id instead would fix this.
- Bag-of-words ignores word order and synonyms. TF-IDF weighting or sentence embeddings would be the next step.

## Credits

- Dataset: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Developed by Aditya Gupta
