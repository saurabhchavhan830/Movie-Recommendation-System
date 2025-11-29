import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Select important features
df['tags'] = df['overview'].fillna('') + " " + df['genres'].astype(str)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
vector = tfidf.fit_transform(df['tags'])

# Compute similarity
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie):
    if movie not in df['title'].values:
        return "Movie not found in database."
    
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("Top 5 Recommendations:")
    for i in movie_list:
        print(df.iloc[i[0]].title)

# Test
recommend("Avatar")
