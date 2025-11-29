# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # merge if possible
    if 'movie_id' in credits.columns and 'id' in movies.columns:
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

    # rename column for title
    if 'title_x' in movies.columns:
        movies = movies.rename(columns={'title_x':'title'})
    elif 'title_y' in movies.columns:
        movies = movies.rename(columns={'title_y':'title'})
    else:
        # if neither exists, fail gracefully
        st.error("No title column found in dataset!")
        return pd.DataFrame()

    movies['overview'] = movies.get('overview', '').fillna('')

    def safe(x):
        if pd.isna(x): return ""
        return str(x)

    movies['genres'] = movies.get('genres', '').apply(safe) if 'genres' in movies.columns else ''
    movies['keywords'] = movies.get('keywords', '').apply(safe) if 'keywords' in movies.columns else ''

    movies['tags'] = (
        movies['overview'].astype(str) + ' ' +
        movies['genres'].astype(str) + ' ' +
        movies['keywords'].astype(str)
    ).str.strip()

    movies.reset_index(drop=True, inplace=True)
    return movies

@st.cache_data(show_spinner=False)
def build_sim(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
    mat = tfidf.fit_transform(texts)
    return cosine_similarity(mat, mat)

movies = load_data()
if not movies.empty:
    sim_matrix = build_sim(movies['tags'].astype(str))
else:
    sim_matrix = None

def get_recommendations(title, top_n=10):
    df = movies
    if df.empty:
        return None
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df['title'].str.lower().str.contains(title.lower(), na=False)]
        if matches.empty:
            return None
    idx = matches.index[0]
    distances = sim_matrix[idx]
    sims = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
    sims = [s for s in sims if s[0] != idx][:top_n]
    return df['title'].iloc[[i for i,_ in sims]].tolist()

st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get similar movie suggestions!")

with st.form("search"):
    movie = st.text_input("Enter a movie title:", "")
    submitted = st.form_submit_button("Recommend")

if submitted and movie.strip():
    out = get_recommendations(movie.strip(), 10)
    if out is None:
        st.warning("Movie not found. Try another name.")
    else:
        st.success("Top Recommendations:")
        for i, m in enumerate(out, 1):
            st.write(f"{i}. {m}")

if 'title' in movies.columns:
    if st.checkbox("Show sample titles"):
        st.write(movies['title'].sample(20).tolist())
