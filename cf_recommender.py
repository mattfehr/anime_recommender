import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

#get data for users ratings
#-1 means unrated
rating_df = pd.read_csv('rating.csv')

#remove possible duplicate rows
rating_df = rating_df.drop_duplicates()

#replace no ratings with NaNs and drop it
rating_df = rating_df.replace(-1, np.NaN)
rating_df = rating_df.dropna(subset=['rating'])

#only include users who have rated at least 100 shows
#get number of ratings from each user
user_counts = rating_df['user_id'].value_counts()

#get users with at least 100 counts
active_users = user_counts[user_counts >= 100].index

#filter out ratings not from qualified users
rating_df = rating_df[rating_df['user_id'].isin(active_users)]

#create user ratings matrix
pivot_matrix = rating_df.pivot(index = 'user_id', columns ='anime_id', values = 'rating')
actual_data = pivot_matrix

#normalize the matrix with centered cosine similarity (Pearson Correlation)
user_means = pivot_matrix.mean(axis=1)
pivot_matrix = pivot_matrix.sub(user_means, axis=0)
pivot_matrix = pivot_matrix.fillna(0)

#convert to sparse matrix
M = pivot_matrix.to_numpy()
M = csr_matrix(pivot_matrix)

#encode user_id and anime_id
#user
#user_ids = rating_df["user_id"].unique()
user_ids = pivot_matrix.index
user_to_encode = {}
encode_to_user = {}
for i, x in enumerate(user_ids):
  user_to_encode[x] = i
  encode_to_user[i] = x
rating_df["user"] = rating_df["user_id"].map(user_to_encode)

#anime
#anime_ids = rating_df["anime_id"].unique()
anime_ids = pivot_matrix.columns
anime_to_encode = {}
encode_to_anime = {}
for i, x in enumerate(anime_ids):
  anime_to_encode[x] = i
  encode_to_anime[i] = x
rating_df["anime"] = rating_df["anime_id"].map(anime_to_encode)

#decompose matrix
U, E, Vt = svds(M, k=100)

#change E to diagnonal matrix
E = np.diag(E)

#combine E and Vt for R = Q * Pt
Pt = np.dot(E, Vt)

#reconstruct matrix with predictions
Q = U
R = np.dot(Q, Pt)

#convert Reconstructed matrix into dataframe
R_df = pd.DataFrame(R, index=user_ids, columns=anime_ids)

#find seen shows
def find_seen_shows(data, user_id):
  seen = set()
  for anime_id, rating in data.loc[user_id].items():
    if rating != 0:
      seen.add(anime_id)
  return seen

#function to get top recommended shows
def find_top_shows(data, R, user_id, top_n=10):

  #sort the predictions of the user from the reconstructed matrix
  sorted_predictions = R.loc[user_id].sort_values(ascending=False)

  #get seen shows
  seen = find_seen_shows(data, user_id)

  #filter out the seen shows
  recommendations = sorted_predictions[~sorted_predictions.index.isin(seen)]
  return recommendations.head(top_n)

#test
print(find_top_shows(pivot_matrix, R_df, 5))