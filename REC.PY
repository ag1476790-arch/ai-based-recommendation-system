import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: Sample Dataset
# -----------------------------
data = {
    'User': ['A','A','A','B','B','C','C','C'],
    'Movie': ['Avengers','Titanic','Inception',
              'Avengers','Inception',
              'Titanic','Inception','Avatar'],
    'Rating': [5,3,4,5,5,4,5,3]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 2: Create User-Movie Matrix
# -----------------------------
user_movie_matrix = df.pivot_table(
    index='User',
    columns='Movie',
    values='Rating'
).fillna(0)

# -----------------------------
# STEP 3: Compute Similarity
# -----------------------------
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# -----------------------------
# STEP 4: Recommendation Function
# -----------------------------
def recommend_movies(user):
    if user not in similarity_df.columns:
        return "User not found!"

    similar_users = similarity_df[user].sort_values(ascending=False)[1:]
    weighted_scores = user_movie_matrix.loc[similar_users.index].T.dot(similar_users)
    recommendations = weighted_scores.sort_values(ascending=False)

    watched_movies = user_movie_matrix.loc[user]
    recommendations = recommendations[watched_movies == 0]

    return recommendations.head(3)

# -----------------------------
# STEP 5: User Input
# -----------------------------
print("Available Users:", list(user_movie_matrix.index))
user_name = input("Enter user name: ")

print("\nRecommended Movies:")
print(recommend_movies(user_name))
