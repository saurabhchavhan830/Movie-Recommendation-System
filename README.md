⭐ Movie Recommendation System (Content-Based Filtering)

A Python-based machine learning project that recommends movies using plot similarity, TF-IDF vectors, and cosine similarity. Built for learning, portfolio, and resume enhancement.

📌 Project Overview

This project builds a Movie Recommendation System using:

🎬 TMDB 5000 Movies Dataset

🧠 TF-IDF Vectorization (Text feature extraction)

📐 Cosine Similarity for measuring similarity between movie plots

🐍 Python and scikit-learn

It recommends movies that are most similar to the movie the user searches.
Perfect skill-project for Data Science, ML, and Python portfolio.

🛠️ Tech Stack
Category	Tools Used
Programming	Python
Machine Learning	scikit-learn (TfidfVectorizer, Cosine Similarity)
Data Handling	Pandas, NumPy
Dataset	TMDB Movies + Credits
Deployment	(Optional) Streamlit
💡 Features

✔ Content-based movie recommendation
✔ Clean TF-IDF preprocessing
✔ High-accuracy cosine similarity search
✔ Fully functional Python script (recommender.py)
✔ Ready for GitHub Portfolio & Resume

📁 Project Structure
Movie-Recommendation-System/
│── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│
│── recommender.py
│── requirements.txt
│── README.md
│── .gitignore

🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/saurabhchavhan830/Movie-Recommendation-System.git
cd Movie-Recommendation-System

2. Install the dependencies
pip install -r requirements.txt

3. Run the recommender
python recommender.py

🧠 How the Model Works
✔ 1. Text Preprocessing

Merging movie and credit data

Selecting important fields

Cleaning overview text

✔ 2. TF-IDF Vectorization

Converts movie overviews into numerical vectors.

✔ 3. Cosine Similarity

Measures distance between these vectors.

✔ 4. Recommendation

Returns top 10 similar movies.

✨ Sample Output
Enter a movie name: Avatar

Recommended Movies:
1. Guardians of the Galaxy
2. John Carter
3. Star Trek
4. Star Wars
5. Avengers
...

🌟 Future Improvements

You can grow this beginner project into a full portfolio ML app:

🟢 Add movie posters (TMDB API)
🟢 Add Streamlit UI
🟢 Add search suggestions
🟢 Add personalised recommendation
🟢 Deploy on Streamlit Cloud

If you want, bro — I can help you upgrade it too 😎🔥

🧑‍💻 Author

Saurabh Chavhan
Beginner Python / C++ / Data Science Learner
GitHub ⭐: https://github.com/saurabhchavhan830