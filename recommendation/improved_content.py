import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from load_data import md, credits, keywords

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md = md.drop([19730, 29503, 35587])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

md['cast'] = md['cast'].apply(literal_eval)
md['crew'] = md['crew'].apply(literal_eval)
md['keywords'] = md['keywords'].apply(literal_eval)
md['cast_size'] = md['cast'].apply(lambda x: len(x))
md['crew_size'] = md['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

md['director'] = md['crew'].apply(get_director)

md['cast'] = md['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['cast'] = md['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

md['keywords'] = md['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

md['cast'] = md['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

md['director'] = md['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
md['director'] = md['director'].apply(lambda x: [x,x, x])

s = md.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()
s = s[s > 1]
stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

md['keywords'] = md['keywords'].apply(filter_keywords)
md['keywords'] = md['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
md['keywords'] = md['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

md['soup'] = md['keywords'] + md['cast'] + md['director'] + md['genres']
md['soup'] = md['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(md['soup'])

md = md.reset_index()
titles = md['title']
indices = pd.Series(md.index, index=md['title'])

# Content Filtering results without taking IMDB votings into account
# def get_recommendations(title):
#     idx = indices[title]
#     cosine_sim = linear_kernel(count_matrix[idx], count_matrix)
#     sim_scores = list(enumerate(cosine_sim[0]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:31]
#     movie_indices = [i[0] for i in sim_scores]
#     return titles.iloc[movie_indices]

# get_recommendations('Inception').head(10)

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

sim_movie_list = {}

def cosine_sim(count_matrix, idx, title):
    if (title in sim_movie_list):
        # print('call from here')
        return sim_movie_list[title]
    else:
        cosine_sim = linear_kernel(count_matrix[idx], count_matrix)
        sim_movie_list[title] = cosine_sim[0]
        return cosine_sim[0]


def improved_recommendations(title):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim(count_matrix, idx, title)))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = md.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.50)
    qualified = movies[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


def weighted_rating(x):
    m = vote_counts.quantile(0.60)
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

print(improved_recommendations('Inception'))
