from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

#get anime data
anime_df = pd.read_csv("anime.csv")
anime_df = anime_df.drop(["type", "members"], axis=1)

#fill empty genres with empty string
anime_df['genre'] = anime_df['genre'].fillna('')

#join titles and genres
def compare_column(x):
  return ''.join(x['name']) + ' ' + ''.join(x['genre'])
anime_df['compare'] = anime_df.apply(compare_column, axis=1)

#term frequency - Inverse Document Frequency
#tf - relative freq of any word in a document by dividing instance with total words
#idf - relative count of document containing term by number of docs divided by docs with term
#importnace of each word if TF * IDK, giving matrix where column is word and row is movie
tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(anime_df['compare'])

#calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#create reverse map of indices and anime titles
indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()

#create recommendation
def get_recommendations(title, cosine_sim=cosine_sim):
  #if title not in indices
  if title not in indices:
    print("Show not found")
    return

  #get index of movie with matching title
  idx = indices[title]

  #get pairwise similiarity scores between this movie and all others
  sim_scores = list(enumerate(cosine_sim[idx]))

  #sort based on sim scores
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  #get 10 most similar indicies
  sim_scores = sim_scores[1:11]
  anime_indices = [i[0] for i in sim_scores]

  #return them
  return anime_df['name'].iloc[anime_indices]

#test
get_recommendations("Kimi no Na wa.", cosine_sim)