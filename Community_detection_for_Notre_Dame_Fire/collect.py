import os
import sys
import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tinydb import TinyDB, where
from multiprocessing import Pool

def read_topics_file(filename):
    with open(filename) as f:
        screen_names = f.read().splitlines()
    return screen_names

def collect_tweets(json_string):
    json_tweet = json.loads(json_string)
    user_name = json_tweet['user']
    screen_name = user_name['screen_name']
    user_id = user_name['id']
    hashtag_info = json_tweet['entities']['hashtags']
    tweet_id = json_tweet['id']
    
    if hashtag_info != []:  
        for ht in hashtag_info:
            hashtags = ht['text']

    
    try :
        text = json_tweet['extended_tweet']['full_text']
    except KeyError as e:
        text = json_tweet['text']

    user_data = {'screen_name':user_name['screen_name'],'user_id':user_name['id']}
    tweet_data = {'text': text,'tweet_id':tweet_id ,'screen_name':screen_name,'user_id':user_id,'created_at':json_tweet['created_at'],'hashtags':hashtag_info}

    return tweet_data,user_data

def start_thread(x):
    stream_obj_dict[x].filter(track=[x])

class listener(StreamListener):
    def __init__(self, db='tweets'):
        self.name=db
        self.tw = TinyDB('./twitter_data/'+db+'.json')
        self.user_db= TinyDB('./twitter_data/user_'+db+'.json')
    
    def on_data(self, json_string):
        tweet_data, user_data = collect_tweets(json_string) #change
        tweet = self.tw.upsert(tweet_data, cond= where('tweet_id') == tweet_data['tweet_id'])
        user = self.user_db.upsert(user_data, cond= where('user_id') == user_data['user_id'])
        print(f'{self.name} - Tweeted by {tweet_data["screen_name"]} for {tweet_data["hashtags"]}')
        print(f'{self.name} - User data stored for {user_data["screen_name"]}')
        return(True)
    
    def on_error(self, status):
        print(self.name,' - Twitter status ',status)
    
    def __del__(self):
        self.tw.close()
        self.user_db.close()

consumer_key = '' #insert API keys here
consumer_secret = '' #insert API keys here
access_token = '' #insert API keys here
access_token_secret = '' #insert API keys here

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

if not os.path.exists('twitter_data'):
    os.makedirs('twitter_data')

print('Starting data collection')
hashtag_list = read_topics_file('topics')
stream_obj_dict = {}
for ht in hashtag_list:
    stream_obj_dict[ht] = Stream(auth, listener(db=ht))

try:
    pool = Pool(len(hashtag_list))
    pool.map(start_thread, hashtag_list)
except KeyboardInterrupt:
    pool.terminate()
finally:
    pool.join()
