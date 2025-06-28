import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Load anime data ---
anime_df = pd.read_csv("anime.csv")
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['rating'] = anime_df['rating'].fillna(0).astype(str)
anime_df['members'] = anime_df['members'].fillna(0).astype(int).astype(str)

# Combine text fields
def build_compare_column(row):
    return f"{row['name']} {row['genre']} {row['type']} rating:{row['rating']} members:{row['members']}"
anime_df['compare'] = anime_df.apply(build_compare_column, axis=1)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['compare'])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Anime ID to index mapping
anime_id_to_idx = pd.Series(anime_df.index, index=anime_df['anime_id']).to_dict()
idx_to_anime_id = pd.Series(anime_df['anime_id'].values, index=anime_df.index).to_dict()

# --- Load user ratings ---
user_df = pd.read_csv("user_ratings.csv")
user_df["rating"] = pd.to_numeric(user_df["rating"], errors="coerce")
user_df = user_df.dropna(subset=["rating"])

# Define what counts as a 'liked' anime
liked_anime_ids = user_df[user_df["rating"] >= 7]["anime_id"].tolist()
seen_anime_ids = set(user_df["anime_id"].tolist())

# --- Compute content-based recommendations ---
def content_based_user_recommendation(liked_ids, seen_ids, top_n=10):
    sim_scores = np.zeros(len(anime_df))

    valid_count = 0
    for anime_id in liked_ids:
        if anime_id in anime_id_to_idx:
            idx = anime_id_to_idx[anime_id]
            sim_scores += cosine_sim[idx]
            valid_count += 1

    if valid_count == 0:
        print("No liked anime found in the dataset.")
        return pd.DataFrame()

    sim_scores /= valid_count
    scores_series = pd.Series(sim_scores, index=anime_df.index)
    scores_series.index = anime_df["anime_id"]

    # Filter out already seen anime
    scores_series = scores_series.drop(labels=seen_ids, errors='ignore')

    top_ids = scores_series.sort_values(ascending=False).head(top_n).index
    results = anime_df[anime_df["anime_id"].isin(top_ids)].copy()
    results["score"] = scores_series.loc[top_ids].values

    return results.sort_values("score", ascending=False)[["anime_id", "name", "genre", "type", "score"]]

# --- Run personalized content-based recommendation ---
recommendations = content_based_user_recommendation(liked_anime_ids, seen_anime_ids, top_n=10)
print(recommendations)
