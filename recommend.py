import numpy as np
import pandas as pd
from numpy.linalg import lstsq

def recommend_for_new_user(
    new_user_ratings,          # Dict[anime_id] = rating
    Vt,                        # Vt from SVD: item factors (shape: k x num_items)
    anime_to_encode,           # Dict[anime_id] -> column index in Vt
    anime_index_to_id,         # Dict[column index] -> anime_id
    top_n=10
):
    # Step 1: Encode ratings to match your Vt item indices
    encoded_ratings = {}
    for anime_id, rating in new_user_ratings.items():
        if anime_id in anime_to_encode:
            idx = anime_to_encode[anime_id]
            encoded_ratings[idx] = rating

    if len(encoded_ratings) == 0:
        print("⚠️ No known anime IDs from new user ratings match the model.")
        return pd.Series(dtype=float)

    # Step 2: Create rating vector and mean-center
    ratings_array = np.zeros(Vt.shape[1])
    for idx, rating in encoded_ratings.items():
        ratings_array[idx] = rating

    rated_indices = list(encoded_ratings.keys())
    mean_rating = np.mean([rating for rating in encoded_ratings.values()])
    centered_ratings = ratings_array.copy()
    centered_ratings[rated_indices] -= mean_rating

    # Step 3: Project user into latent space using least squares
    Vt_rated = Vt[:, rated_indices]  # shape: (k x num_rated)
    r_rated = centered_ratings[rated_indices]  # shape: (num_rated,)
    q_new, _, _, _ = lstsq(Vt_rated.T, r_rated, rcond=None)  # shape: (k,)

    # Step 4: Predict ratings for all items
    predicted_ratings = np.dot(q_new, Vt) + mean_rating

    # Step 5: Build output series and remove already rated
    pred_series = pd.Series(predicted_ratings, index=[anime_index_to_id[i] for i in range(Vt.shape[1])])
    pred_series = pred_series.drop(labels=new_user_ratings.keys(), errors='ignore')

    # Step 6: Return top N
    return pred_series.sort_values(ascending=False).head(top_n)
