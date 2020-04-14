from load_data import movies_df, movies_metadata_df
from sklearn.metrics.pairwise import linear_kernel
import pickle
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load keywords and credits
credits = pd.read_csv('../data/credits.csv')
keywords = pd.read_csv('../data/keywords.csv')

credits = credits.drop_duplicates(subset=['id'])
keywords = keywords.drop_duplicates(subset=['id'])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies_metadata_df['id'] = movies_metadata_df['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
movies_metadata_df = credits.merge(movies_metadata_df, on='id')
movies_metadata_df = movies_metadata_df.merge(keywords, on='id')


movies_df = pd.Series(movies_metadata_df.index, index=movies_metadata_df['title']).drop_duplicates()

# features = ['cast', 'crew', 'keywords', 'genres']
# for feature in features:
#     movies_metadata_df[feature] = movies_metadata_df[feature].apply(literal_eval)

features = ['cast', 'crew', 'keywords']
for feature in features:
    movies_metadata_df[feature] = movies_metadata_df[feature].apply(literal_eval)

movies_metadata_df['genres'] = movies_metadata_df['genres'].fillna('[]')
movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(lambda x: literal_eval(str(x)))


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Define new director, cast, genres and keywords features that are in a suitable form.
movies_metadata_df['director'] = movies_metadata_df['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    movies_metadata_df[feature] = movies_metadata_df[feature].apply(get_list)

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    movies_metadata_df[feature] = movies_metadata_df[feature].apply(clean_data)

# Create a new soup feature
movies_metadata_df['soup'] = movies_metadata_df.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies_metadata_df['soup'])
#count_matrix
cosine_sim = linear_kernel(count_matrix[0], count_matrix)

def get_recommendations(title, tfidf_matrix):
    # Get the index of the movie that matches the title
    idx = movies_df[title]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[0]))

    pickle.dump(tfidf_matrix, open("../feature/feature.pkl", "wb"))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:21]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    # print(idx, title , movies_metadata_df['title'].iloc[idx],movies_metadata_df['genres'].iloc[idx])
    return movies_metadata_df['title'].iloc[movie_indices]

#Find movies similar to 'Toy Story' using Content Filtering
print(get_recommendations('Toy Story',count_matrix))