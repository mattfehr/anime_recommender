import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# --- Load anime data ---
anime_df = pd.read_csv("anime.csv")
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['rating'] = anime_df['rating'].fillna(0).astype(str)
anime_df['members'] = anime_df['members'].fillna(0).astype(int).astype(str)

# Combine text fields for content-based filter
def build_compare_column(row):
    return f"{row['name']} {row['genre']} {row['type']} rating:{row['rating']} members:{row['members']}"
anime_df['compare'] = anime_df.apply(build_compare_column, axis=1)

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['compare'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Anime ID to index mapping
anime_id_to_idx = pd.Series(anime_df.index, index=anime_df['anime_id']).to_dict()

# --- Load and process user ratings ---
rating_df = pd.read_csv('rating.csv')
new_user_df = pd.read_csv('user_ratings.csv')
rating_df = pd.concat([rating_df, new_user_df], ignore_index=True)
rating_df = rating_df.replace(-1, np.NaN).dropna(subset=['rating'])

# Keep active users and the new user
user_counts = rating_df['user_id'].value_counts()
active_users = user_counts[user_counts >= 100].index
new_user_id = new_user_df['user_id'].iloc[0]
rating_df = rating_df[(rating_df['user_id'].isin(active_users)) | (rating_df['user_id'] == new_user_id)]

# Collaborative filtering matrix and SVD
pivot_matrix = rating_df.pivot(index='user_id', columns='anime_id', values='rating')
user_means = pivot_matrix.mean(axis=1)
normalized_matrix = pivot_matrix.sub(user_means, axis=0).fillna(0)
M = csr_matrix(normalized_matrix.values)
U, S, Vt = svds(M, k=100)
S = np.diag(S)
R = np.dot(np.dot(U, S), Vt)
R_df = pd.DataFrame(R, index=pivot_matrix.index, columns=pivot_matrix.columns)

# --- Hybrid Recommendation ---
def hybrid_recommendation(user_id, R_df, anime_df, cosine_sim, anime_id_to_idx, user_df, top_n=10, alpha=0.5):
    # Collaborative scores
    if user_id not in R_df.index:
        print(f"User {user_id} not found in collaborative matrix.")
        return pd.DataFrame()
    collab_scores = R_df.loc[user_id].copy().dropna()

    # Content-based scores based on liked anime
    user_df["rating"] = pd.to_numeric(user_df["rating"], errors="coerce")
    user_df = user_df.dropna(subset=["rating"])
    liked_anime_ids = user_df[user_df["rating"] >= 7]["anime_id"].tolist()
    seen_anime_ids = set(user_df["anime_id"].tolist())

    indices = [anime_id_to_idx[aid] for aid in liked_anime_ids if aid in anime_id_to_idx]
    if not indices:
        print("No valid liked anime found for content-based filtering.")
        return pd.DataFrame()

    content_scores = cosine_sim[indices].mean(axis=0)
    content_series = pd.Series(content_scores, index=anime_df.index)
    content_series.index = anime_df['anime_id']

    # Normalize both score sets
    scaler = MinMaxScaler()
    collab_scores_scaled = pd.Series(
        scaler.fit_transform(collab_scores.values.reshape(-1, 1)).flatten(),
        index=collab_scores.index
    )
    content_series_filled = content_series.reindex(collab_scores_scaled.index).fillna(0)
    content_scores_scaled = pd.Series(
        scaler.fit_transform(content_series_filled.values.reshape(-1, 1)).flatten(),
        index=content_series_filled.index
    )

    # Combine scores
    hybrid_scores = alpha * collab_scores_scaled + (1 - alpha) * content_scores_scaled
    hybrid_scores = hybrid_scores.drop(labels=seen_anime_ids, errors='ignore')

    top_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index
    results = anime_df[anime_df['anime_id'].isin(top_ids)].copy()
    results['score'] = hybrid_scores.loc[top_ids].values

    return results.sort_values("score", ascending=False)[["anime_id", "name", "genre", "type", "score"]]

# --- Run hybrid recommendation ---
print(f"\n Top hybrid recommendations for user '{new_user_id}':")
print(hybrid_recommendation(new_user_id, R_df, anime_df, cosine_sim, anime_id_to_idx, new_user_df, top_n=10, alpha=0.9))
