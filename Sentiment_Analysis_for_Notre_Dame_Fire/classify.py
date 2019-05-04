import re
import numpy as np
import string
import json
import nltk
import pickle
from collections import defaultdict, Counter
from tinydb import TinyDB
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC

def dump_data(summary, pick = 'data_from_classifypy'):
    with open(pick,'wb') as f:
        pickle.dump(summary,f)

def tweet_cleaner(tweet):
    punctuation = string.punctuation
    tweet = re.sub(r'(@[A-Za-z0-9_]+)', '' , tweet.lower())
    tweet = re.sub(r'\&\w*;', '', tweet.lower())
    tweet = re.sub(r'\$\w*', '', tweet.lower())
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet.lower())
    tweet = re.sub(r'#\w*', '', tweet.lower())
    tweet = re.sub(r'^RT[\s]+', '', tweet.lower())
    tweet = ''.join(c for c in tweet.lower() if c <= '\uFFFF')
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet.lower())
    return tweet

def tokenize(tweet, keep_internal_punct=True):
    tweet_l = tweet.lower()
    token = []
    if keep_internal_punct:
        for word in tweet_l.split():
            token.append(word.strip(string.punctuation))
    else:
        token = re.findall(r"\w+", doc_l)
    return token

def lemmatize(tweet):
    lemmatizer = WordNetLemmatizer()
    lemmatized_token = []
    for word in tweet:
        lemmatized_token.append(lemmatizer.lemmatize(word))
    lemmatized_token = list(filter(None, lemmatized_token))
    return lemmatized_token

def preprocess(path, labels):
    with open(path) as f:
        json_file = json.load(f)
    text_dict = defaultdict(list)
    for tweet in json_file:
        tweet_dict = dict(tweet)
        text_dict[tweet_dict['label']].append(categorize(tweet_dict['text'])[0])
    #     for value in labels:
    #         text_dict[value] = lemmatize(text_dict[value])
    return text_dict

def categorize(text):
    clean_tweet = tweet_cleaner(text)
    tweet_token = tokenize(clean_tweet)
    lemmatized_tweet = lemmatize(tweet_token)
    tweet_token = list(filter(None, lemmatized_tweet))
    return clean_tweet,tweet_token

def split_data(tweets_pos_tokens, tweets_neg_tokens, tweets_other_tokens, training_samples=1000):
    y_train = (['pos'] * training_samples)+(['neg']* training_samples)+(['other']* training_samples)
    y_test = ['pos'] * (len(tweets_pos_tokens)-training_samples)+(['neg']* (len(tweets_neg_tokens)-training_samples)) + (['other'] * (len(tweets_other_tokens)-training_samples))
    X_train = tweets_pos_tokens[:training_samples]+tweets_neg_tokens[:training_samples]+tweets_other_tokens[:training_samples]
    X_test = tweets_pos_tokens[training_samples:]+tweets_neg_tokens[training_samples:]+tweets_other_tokens[training_samples:]
    return X_train,y_train,X_test,y_test

def classifier(X_train,y_train,vectorizer):
    X = vectorizer.fit_transform(X_train)
    clf = SVC(probability = True)
    clf.fit(X, y_train)
    print("n_samples: %d, n_features: %d" % X.shape)
    return clf

labels = ['pos', 'neg', 'other']
text = preprocess('training.json', labels)
X_train,y_train,X_test,y_test = split_data(text['pos'], text['neg'], text['other'], 100)

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=2)
clf = classifier(X_train,y_train,vectorizer)
X_test_vect = vectorizer.transform(X_test)
predictions = clf.predict(X_test_vect)
counter_state = Counter(predictions)

summary = []
example = dict()
summary.append(' ')
summary.append('Instance count for each label :')
summary.append(' ')
summary.append('--pos : instance count for positive labels, --neg : instance count for positive labels')
summary.append('--other : instance count for tweets that have neither positive nor negative sentiment.')
summary.append(' ')
for class_instance, count in counter_state.items():
    summary.append(f'{class_instance} : {count}')
summary.append(' ')
summary.append('Examples for each class :')
summary.append(' ')
summary.append('--pos : example for positive labels, --neg : example for positive label')
summary.append('--other: example for tweet that has neither negative nor positive sentiment')
summary.append(' ')
for i,label in enumerate(predictions):
    if label not in example:
        example[predictions[i]] = X_test[i]
        summary.append(f'{label} : {"".join(X_test[i])}')
dump_data(summary)
