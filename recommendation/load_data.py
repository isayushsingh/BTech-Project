import pandas as pd
import numpy as np
from utils import convert_ids, convert_ids, convert_to_float, to_json
movies_metadata_df1 = pd.read_csv('../data/movies_metadata.csv'
                                 , converters={ 'id': lambda x: convert_ids(x)
                                               , 'imdb_id': lambda x: convert_ids(x)
                                               ,'popularity': lambda x: convert_to_float(x)
                                               ,'genres': lambda x: to_json(x)}
                                 , usecols=['id', 'original_title'
                                                , 'genres' #'homepage'
                                                , 'overview', 'popularity', 'poster_path'
                                                , 'release_date', 'revenue', 'runtime'
                                                , 'spoken_languages', 'title'
                                                , 'vote_average', 'vote_count']
                                , dtype={'populariy': np.float64}
                                , parse_dates=True, low_memory=False)


movies_lookup_df = pd.read_csv('../data/movies_metadata.csv'
                        , converters={'id': lambda x: convert_ids(x), 'imdb_id': lambda x: convert_ids(x)}
                       ,usecols=['id', 'title'], low_memory=False)

#####################################
##SVD DATA SET
movies_df = pd.read_csv('../data/movies_metadata.csv'
                        , converters={'id': lambda x: convert_ids(x), 'imdb_id': lambda x: convert_ids(x)}
                       ,usecols=['id', 'original_title', 'belongs_to_collection'
                                 , 'budget', 'genres', 'homepage'
                                 ,'imdb_id', 'overview', 'popularity', 'poster_path'
                                 , 'production_companies','release_date', 'revenue', 'runtime',
                                 'spoken_languages', 'status', 'tagline', 'title', 'video',
                                 'vote_average', 'vote_count'], low_memory=False)
#####################################

ratings_df = pd.read_csv('../data/ratings_small.csv', low_memory=False)

###May need Fuzzy matching, but for now:
movies_df = movies_df[movies_df.spoken_languages == """[{'iso_639_1': 'en', 'name': 'English'}]"""]

ratings_with_movie_names = ratings_df.merge(movies_df[['id', 'original_title']], how='left', left_on='movieId', right_on='id')
ratings_with_movie_names = ratings_with_movie_names[ratings_with_movie_names.original_title.isnull() == False]



#Loading data for content filtering
movies_metadata_df = pd.read_csv('../data/movies_metadata.csv'
                                 , converters={ 'id': lambda x: convert_ids(x)
                                               , 'imdb_id': lambda x: convert_ids(x)
                                               ,'popularity': lambda x: convert_to_float(x)
                                               ,'genres': lambda x: to_json(x)}
                                 , usecols=['id', 'original_title'
                                                , 'genres' #'homepage'
                                                , 'overview', 'popularity', 'poster_path'
                                                , 'release_date', 'revenue', 'runtime'
                                                , 'spoken_languages', 'title'
                                                , 'vote_average', 'vote_count']
                                , dtype={'populariy': np.float64}
                                , parse_dates=True, low_memory=False)
movies_metadata_df = movies_metadata_df.drop_duplicates(subset=['id'])