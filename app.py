# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # merge files
    if 'movie_id' in credits.columns and 'id' in movies.columns:
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

    # Fix missing title column
    if "title" not in movies.columns:
        if "title_x" in movies.columns:
            movies.rename(columns={"title_x": "title"}, inplace=True)
        elif "title_y" in movies.columns:
            movies.rename(columns={"title_y": "title"}, inplace=True)

    # ensure overview exists
    if "overview" not in movies.columns:
        movies["overview"] = ""
    else:
        movies["overview"] = movies["overview"].fillna("")

    # safe text for genres/keywords
    def safe(x):
        return "" if pd.isna(x) else str(x)

    movies["genres"] = movies["genres"].apply(safe) if "genres" in movies.columns else ""
    movies["keywords"] = movies["keywords"].apply(safe) if "keywords" in movies.columns else ""

    # create tags
    movies["tags"] = (
        movies["overview"].astype(str)
        + " "
        + movies["genres"].astype(str)
        + " "
        + movies["keywords"].astype(str)
    ).str.strip()

    movies.reset_index(drop=True, inplace=True)

    return movies


# -----------------------------------------
# BUILD SIMILARITY MATRIX
# -----------------------------------------
@st.cache_data(show_spinner=False)
def build_sim(texts):
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    mat = tfidf.fit_transform(texts)
    return cosine_similarity(mat, mat)


movies = load_data()
sim_matrix = build_sim(movies["tags"].astype(str))

# -----------------------------------------
# RECOMMENDER FUNCTION
# -----------------------------------------
def get_recommendations(title, top_n=10):
    df = movies

    # exact match
    exact = df[df["title"].str.lower() == title.lower()]

    # partial match fallback
    if exact.empty:
        exact = df[df["title"].str.lower().str.contains(title.lower())]

    if exact.empty:
        return None

    idx = exact.index[0]
    distances = sim_matrix[idx]

    sims = list(enumerate(distances))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    sims = [s for s in sims if s[0] != idx][:top_n]

    return df["title"].iloc[[i for i, _ in sims]].tolist()


# -----------------------------------------
# UI
# -----------------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get similar movies!")

with st.form("search"):
    movie = st.text_input("Enter a movie title:", "")
    submit = st.form_submit_button("Recommend")

if submit and movie.strip():
    out = get_recommendations(movie.strip(), 10)

    if out is None:
        st.warning("Movie not found. Try another name.")
    else:
        st.success("Top Recommendations:")
        for i, m in enumerate(out, 1):
            st.write(f"{i}. {m}")

# optional testing
if st.checkbox("Show sample titles"):
    st.write(movies["title"].sample(20).tolist())
