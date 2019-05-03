# -*- coding: utf-8 -*-

# coding: utf-8

# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()

def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())

def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    token_list = []
    l = []
    for element in movies['genres']:
        if movies['genres'][1] != "(no genres listed)":
            token_list.append(tokenize_string(element))
        else:
            token_list.append(l)

    movies['tokens'] = token_list
    return movies
  
  
def featurize(movies): 
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """ 
    ###TODO
    #number of movies
    N = len(movies)
    #list to store values of csr matrix
    csr_list = []
    token_list_ol = movies['tokens'].tolist()
    #number of unique genres sorted
    uniq_token_list = set()
    #genre and their frequency in entire column
    df_list=[]

    for element in token_list_ol:
        for value in element:
            if value not in uniq_token_list:
                #add genre to set if not present already 
                uniq_token_list.add(value)
                
    #calculation for df
    for x in movies['tokens']:
        for i in set(x):
            df_list.append(i)
    doc_freq_c = Counter(df_list)
    doc_freq_c = dict(doc_freq_c) 
#print (doc_freq_c)

    #sorting the list and getting count of total number of features
    uniq_token_list = sorted(uniq_token_list)
    N = len(movies)

    #assigning index to each genre and forming a vocab
    vocab = {}
    for value in uniq_token_list:
        vocab[value] = len(vocab)
    num_features = len(vocab)

    #calculating tf, df and forming csr matrix
    for sublist in token_list_ol:
        #initilizations for csr calculations
        tfidf_list = []
        row_indices = []
        row_ptr = []
        #calculate tf
        tf_dict = defaultdict(lambda : 0)
        for value in sublist:
            if value not in tf_dict:
                tf_dict[value] = 1
            else:
                tf_dict[value] += 1
        
        for value in tf_dict:
            #calculate tf-idf values
            tf = tf_dict[value]
            df = doc_freq_c[value]
            max_k = max(tf_dict.values())
            tfidf = (tf/max_k)* math.log10(N/df)
            tfidf_list.append(tfidf)
        #calculate csr
            index = vocab[value]
            row_indices.append(index)

        row_ptr = [0,len(row_indices)]
        csr_list.append(csr_matrix((tfidf_list,row_indices,row_ptr),shape=(1,num_features)))
    
    movies['features'] = csr_list
    return(movies,vocab)
        

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    a_l2 = np.linalg.norm(a.toarray())
    b_l2 = np.linalg.norm(b.toarray())
    denominator = a_l2 * b_l2
    numerator = a.dot(b.transpose()).toarray()[0][0]
    return numerator/denominator

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    rating_predictions = []
    for index,row in ratings_test.iterrows():
      cosine_sim_list = []
      weighted_ratings = []
      userid_test = row['userId']
      movieid_test = row['movieId']
      csr_test = movies[movies['movieId'] == movieid_test]['features'].iloc[0]

      for index_train,row_train in ratings_train[ratings_train.userId == userid_test].iterrows():
          movieid_train = row_train['movieId']
          csr_train = movies[movies['movieId'] == movieid_train]['features'].iloc[0]
          cosine_val = cosine_sim(csr_test,csr_train)
          if cosine_val > 0.0:
              cosine_sim_list.append(cosine_val)
              weighted_ratings.append(row_train['rating'])

      if cosine_sim_list:
          rating_predictions.append(np.dot(cosine_sim_list, weighted_ratings) / sum(cosine_sim_list))
      else:
          rating_predictions.append(ratings_train[ratings_train.userId == userid_test].rating.mean())

    rating_predictions = np.array(rating_predictions)
    return rating_predictions 

def mean_absolute_error(predictions, ratings_test):
    """DONE.ting
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()

def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
#	import doctest
#	doctest.testmod()

