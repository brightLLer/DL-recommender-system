import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def read_movielens_data(path='./ml-1m/ratings.dat', test_size=0.1, random_state=42):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_table(path, names=r_cols, sep = '::', engine = 'python')
    ratings_train, ratings_test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    num_users, num_items = ratings['user_id'].max(), ratings['movie_id'].max()
    movie_lens = {
        "original": ratings,
        "train_set": ratings_train,
        "test_set": ratings_test,
        "num_users": num_users,
        "num_items": num_items
    }
    return movie_lens

def create_user_item_matrix(dataset, num_users, num_items):
    data_matrix = np.zeros((num_users, num_items))
    for row in dataset.itertuples():
        data_matrix[row[1]-1, row[2]-1] = row[3]
    return data_matrix