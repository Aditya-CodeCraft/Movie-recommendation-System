# Movie-recommendation-System
"Movie Mate" is a web application that solves 'choice paralysis' problem. A user selects a movie they like, and the system instantly recommends five similar movies based on their content (genre, plot, cast, etc.), simplifying the discovery process.


Here’s a polished and styled `README.md` in Markdown format for your project **Your Movie Mate**. It includes setup instructions, data processing steps, model saving, and GUI integration.

---

# 🎬 Your Movie Mate

Your Movie Mate is a content-based movie recommendation system powered by TMDB movie metadata. It analyzes movie features like cast, crew, genres, and keywords to suggest similar titles based on your selection. With a simple GUI, it's your personal movie companion!

---

## 📁 Dataset

Download the following datasets from Kaggle:

🔗 [TMDB Movie Metadata Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data)

Place these files in your project directory:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

---

## ⚙️ Setup & Preprocessing

Create a Python file (e.g., `main.py`) and follow these steps:

### 1. 📦 Import Libraries

```python
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### 2. 📥 Load Datasets

```python
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```

### 3. 🔗 Merge Datasets

```python
movies = movies.merge(credits, on='title')
```

### 4. 🧹 Select & Clean Features

```python
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

### 5. 🔍 Parse JSON-like Columns

```python
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies['crew'] = movies['crew'].apply(fetch_director)
```

### 6. 🧠 Create Tags Feature

```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['crew'] = movies['crew'].apply(lambda x: [x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
```

### 7. 📊 Vectorization & Similarity

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
```

### 8. 💾 Save Model with Pickle

```python
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

---

## 🖥️ GUI Integration

Create a file named `apps.py` and use the saved `.pkl` files to build a simple GUI. You can use `streamlit`, `tkinter`, or `flask` depending on your preference.

### Example with Streamlit:

```python
import streamlit as st
import pickle

movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies = list(movie_dict['title'].values())

def recommend(movie):
    index = movies.index(movie)
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies[i[0]] for i in movie_list]

st.title("🎬 Your Movie Mate")
selected_movie = st.selectbox("Choose a movie", movies)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("You might also like:")
    for title in recommendations:
        st.write(f"👉 {title}")
```

---

## 🚀 Run the App

```bash
streamlit run apps.py
```

---

## ✅ Features

- Content-based filtering using cast, crew, genres, keywords, and overview
- Cosine similarity for recommendations
- GUI for interactive movie selection
- Pickle-based model persistence

---

## 📌 Requirements

Install dependencies using:

```bash
pip install pandas numpy scikit-learn streamlit
```

---

## 🧠 Credits

- Dataset: [TMDB on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Developed by: *Aditya Gupta*

---
