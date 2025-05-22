
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# --- Load Anime Metadata ---
anime_df = pd.read_csv("anime.csv")
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['rating'] = anime_df['rating'].fillna(0).astype(str)
anime_df['members'] = anime_df['members'].fillna(0).astype(int).astype(str)

# Build text feature for content-based filtering
def build_compare_column(row):
    return f"{row['name']} {row['genre']} {row['type']} rating:{row['rating']} members:{row['members']}"
anime_df['compare'] = anime_df.apply(build_compare_column, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(anime_df['compare'])

# Cosine Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Index mappings
anime_id_to_idx = pd.Series(anime_df.index, index=anime_df['anime_id']).to_dict()
idx_to_anime_id = pd.Series(anime_df['anime_id'].values, index=anime_df.index).to_dict()

# --- Load Ratings and Reconstruct Collaborative Matrix ---
rating_df = pd.read_csv('rating.csv')
rating_df = rating_df.replace(-1, np.NaN).dropna(subset=['rating'])

pivot_matrix = rating_df.pivot(index='user_id', columns='anime_id', values='rating')
user_means = pivot_matrix.mean(axis=1)
normalized_matrix = pivot_matrix.sub(user_means, axis=0).fillna(0)

M = csr_matrix(normalized_matrix.values)
U, S, Vt = svds(M, k=100)
S = np.diag(S)
R = np.dot(np.dot(U, S), Vt)
R_df = pd.DataFrame(R, index=pivot_matrix.index, columns=pivot_matrix.columns)

# --- Hybrid Recommendation Function ---
def hybrid_recommendation(user_id, R_df, anime_df, cosine_sim, anime_id_to_idx, top_n=10, alpha=0.5):
    if user_id not in R_df.index:
        print(f"User {user_id} not found in collaborative matrix.")
        return pd.DataFrame()

    # Collaborative predictions
    collab_scores = R_df.loc[user_id].copy().dropna()
    top_collab_ids = collab_scores.sort_values(ascending=False).head(30).index

    # Aggregate content similarity
    content_scores = np.zeros(len(anime_df))
    for anime_id in top_collab_ids:
        if anime_id in anime_id_to_idx:
            idx = anime_id_to_idx[anime_id]
            content_scores += cosine_sim[idx]

    content_scores /= len(top_collab_ids)
    content_series = pd.Series(content_scores, index=anime_df.index)
    content_series.index = anime_df['anime_id']

    # Combine scores
    hybrid_scores = alpha * collab_scores.add((1 - alpha) * content_series, fill_value=0)

    # Remove already seen
    seen = R_df.loc[user_id][R_df.loc[user_id] > 0].index
    hybrid_scores = hybrid_scores.drop(seen, errors="ignore")

    # Top-N
    top_ids = hybrid_scores.sort_values(ascending=False).head(top_n).index
    results = anime_df[anime_df['anime_id'].isin(top_ids)].copy()
    results['score'] = hybrid_scores.loc[top_ids].values
    return results.sort_values("score", ascending=False)[["anime_id", "name", "genre", "type", "score"]]

# --- Example Test ---
print(hybrid_recommendation(user_id=5, R_df=R_df, anime_df=anime_df, cosine_sim=cosine_sim, anime_id_to_idx=anime_id_to_idx, top_n=10, alpha=0.7))
