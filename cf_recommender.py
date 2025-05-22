import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# --- Load data ---
ratings_df = pd.read_csv('rating.csv')
new_user_df = pd.read_csv('user_ratings.csv')  # one MAL user

# --- Append new user ratings ---
ratings_df = pd.concat([ratings_df, new_user_df], ignore_index=True)

# --- Clean and preprocess ---
ratings_df = ratings_df.drop_duplicates()
ratings_df['rating'] = ratings_df['rating'].replace(-1, np.NaN)
ratings_df = ratings_df.dropna(subset=['rating'])

# --- Filter active users (exclude new user for this filter) ---
user_counts = ratings_df['user_id'].value_counts()
active_users = user_counts[user_counts >= 100].index

# Keep active users or the new user
new_user_id = new_user_df['user_id'].iloc[0]
ratings_df = ratings_df[(ratings_df['user_id'].isin(active_users)) | (ratings_df['user_id'] == new_user_id)]

# --- Create user-item matrix ---
pivot_matrix = ratings_df.pivot(index='user_id', columns='anime_id', values='rating')
actual_data = pivot_matrix.copy()

# --- Normalize (centered cosine / Pearson) ---
user_means = pivot_matrix.mean(axis=1)
pivot_matrix = pivot_matrix.sub(user_means, axis=0).fillna(0)

# --- Sparse matrix and SVD ---
M = csr_matrix(pivot_matrix.to_numpy())
U, E, Vt = svds(M, k=100)
E = np.diag(E)

# --- Reconstruct predicted matrix ---
Q = U
Pt = np.dot(E, Vt)
R = np.dot(Q, Pt)

# --- Convert to DataFrame ---
R_df = pd.DataFrame(R, index=pivot_matrix.index, columns=pivot_matrix.columns)

# --- Recommendation logic ---
def find_seen_shows(data, user_id):
    return set(data.loc[user_id][data.loc[user_id] != 0].index)

def find_top_shows(data, R, user_id, top_n=10):
    seen = find_seen_shows(data, user_id)
    sorted_predictions = R.loc[user_id].sort_values(ascending=False)
    recommendations = sorted_predictions[~sorted_predictions.index.isin(seen)]
    return recommendations.head(top_n)

# --- Recommend for new user ---
print(f"\n🎯 Top recommendations for user '{new_user_id}':")
print(find_top_shows(pivot_matrix, R_df, new_user_id))
