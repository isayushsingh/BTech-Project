import pandas as pd
from collections import defaultdict
import json
import re

def convert_ids(ids_in_csv):
    return pd.to_numeric(ids_in_csv, errors='coerce').astype('int64')

def convert_to_float(ids_in_csv):
    return pd.to_numeric(ids_in_csv, errors='coerce').astype('float64')

def to_json(csv_entry):
    return json.loads(re.sub('\'', '"', csv_entry))


def get_top_n(predictions, n=200):
    '''SUPRISE API
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_movie_name(movie_id):
    return ratings_with_movie_names[ratings_with_movie_names.id == movie_id]['title'].iloc[0]


def print_user_prediction(userId, predictions_dict, meta_df):
    users_viewed_movies = ratings_with_movie_names[ratings_with_movie_names['userId'] == userId][
        ['rating', 'original_title']]
    print(f'User {userId} has viewed the following movies:\n')

    for row in users_viewed_movies.itertuples():
        rating = row[1]
        original_title = row[2]
        print(f'\t{original_title}, Rating: {rating}')

    print(f'\nThe following movies are recommended for User {userId}\n')
    recommended_movies = [get_movie_name(mov_id[0], meta_df) for mov_id in predictions_dict[userId]]

    for movie in recommended_movies:
        print(f'\t{movie}')

def get_movie_name(movie_id, movie_meta_df):
    return movie_meta_df[movie_meta_df.id == movie_id]['title'].iloc[0]

def get_movie_id(title, movie_meta_df):
    return movie_meta_df[movie_meta_df.title == title]['id'].iloc[0]


# def get_all_movies_in_cluster(cluster_number, cluster_dict, meta_df):
#     movies = cluster_dict[cluster_number]
#     return [get_movie_name(mov, meta_df) for mov in movies]
#
# def get_cluster_number(movie, cluster_zip):
#     for cluster, movie_id in cluster_zip:
#
#         if movie_id == movie:
#             return cluster
#
#     raise Exception('Movie not found in cluster')