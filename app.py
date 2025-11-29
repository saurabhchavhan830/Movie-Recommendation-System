# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_data(show_spinner=False)
def load_data():
    # adjust filenames if yours are named differently
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    # try to merge on id / movie_id if both files present
    if 'movie_id' in credits.columns and 'id' in movies.columns:
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    movies['overview'] = movies['overview'].fillna('')
    # Create a single text field to vectorize (overview + genres + keywords if available)
    def safe_text(x):
        if pd.isna(x):
            return ""
        return str(x)
    movies['genres'] = movies.get('genres', '').apply(safe_text) if 'genres' in movies.columns else ''
    movies['keywords'] = movies.get('keywords', '').apply(safe_text) if 'keywords' in movies.columns else ''
    movies['tags'] = (movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords']).str.strip()
    movies.reset_index(drop=True, inplace=True)
    return movies

@st.cache_data(show_spinner=False)
def build_sim_matrix(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
    matrix = tfidf.fit_transform(texts)
    sim = cosine_similarity(matrix, matrix)
    return sim

movies = load_data()
sim_matrix = build_sim_matrix(movies['tags'].astype(str))

def get_recommendations(title, top_n=10):
    # case-insensitive exact match first, then contains fallback
    df = movies
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df['title'].str.lower().str.contains(title.lower())]
        if matches.empty:
            return None
    idx = matches.index[0]
    distances = sim_matrix[idx]
    sims = list(enumerate(distances))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [s for s in sims if s[0] != idx][:top_n]
    return df['title'].iloc[[i for i,_ in sims]].tolist()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
st.markdown("Type a movie name and get similar movie suggestions (content-based using TF-IDF + Cosine Similarity).")

with st.form("search_form"):
    movie_input = st.text_input("Enter a movie title (e.g. Avatar)", "")
    submitted = st.form_submit_button("Recommend")

if submitted and movie_input.strip():
    result = get_recommendations(movie_input.strip(), top_n=10)
    if result is None:
        st.warning("Movie not found in dataset. Try another title (partial names work).")
    else:
        st.success("Top recommendations:")
        for i, m in enumerate(result, start=1):
            st.write(f"{i}. {m}")

# optional: show a few sample titles for quick tests
if st.checkbox("Show sample movie titles (for quick testing)"):
    st.write(movies['title'].sample(20).tolist())
