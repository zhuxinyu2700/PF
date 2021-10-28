import numpy as np
import pandas as pd

def make_dataset_1M(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=r_cols,encoding='latin-1')
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings)*0.8))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    if load_sidechannel:
        u_cols = ['user_id','sex','age','occupation','zip_code']
        m_cols = ['movie_id','title','genre']
        users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        movies = pd.read_csv('./ml-1m/movies.dat', sep='::', names=m_cols,
                            encoding='latin-1', parse_dates=True)

    train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    columnsTitles=["user_id","rating","movie_id"]
    train_ratings=train_ratings.reindex(columns=columnsTitles)-1
    test_ratings=test_ratings.reindex(columns=columnsTitles)-1
    users.user_id = users.user_id.astype(np.int64)
    movies.movie_id = movies.movie_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1



    if load_sidechannel:
        return train_ratings,test_ratings,users,movies
    else:
        return train_ratings,test_ratings


