from surprise import SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise import Dataset
from surprise import Reader
from collections import defaultdict

from load_data import ratings_with_movie_names, movies_metadata_df1
from utils import get_top_n, get_movie_name

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings_with_movie_names[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algo = SVD(verbose=True)
algo.fit(trainset)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)

predictions = algo.test(testset)
### Tune this value to get fewer results faster, but less options to choose from
top_n = get_top_n(predictions)


predicted_movies_by_name = defaultdict(list)

### This builds the dictionary of predicted movies for all users
for key, value in top_n.items():
    predicted_movies_by_name[key] = [get_movie_name(mov_id[0], movies_metadata_df1) for mov_id in value]

from collections import namedtuple

UserFavoriteRating = namedtuple('UserFavoriteRating', ['title', 'rating'])


def users_top_n_movies(n, userId, predictions_dict, meta_df):
    users_viewed_movies = ratings_with_movie_names[ratings_with_movie_names['userId'] == userId][
        ['rating', 'original_title']]

    viewed_movies = []

    for row in users_viewed_movies.itertuples():
        rating = row[1]
        original_title = row[2]
        film = UserFavoriteRating(original_title, rating)
        viewed_movies.append(film)

    sorted(viewed_movies, key=lambda film: film[1])

    return viewed_movies[0:n]

#users_top_n_movies(6, 10, predicted_movies_by_name, movies_metadata_df1)
#print_user_prediction(47, top_n, movies_metadata_df1)

UserFavoriteRating = namedtuple('UserFavoriteRating', ['title', 'rating'])

def collab_filter_recommendations(user, top_ns, movie_meta_df):
    predictions = top_ns[user]

    return [UserFavoriteRating(get_movie_name(pred[0], movie_meta_df), pred[1]) for pred in predictions]

#Print Collaborative Filter Recommmendations for user 47
print(collab_filter_recommendations(47, top_n, movies_metadata_df1))