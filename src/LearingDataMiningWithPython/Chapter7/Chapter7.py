#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/20 14:13
# @Author  : Aries
# @Site    : 
# @File    : Chapter7.py
# @Software: PyCharm

import twitter

consumer_key = "GUK5q674gkJDcRh6syRatOcth"
consumer_secret = "LMBcY7nHkJfQWjZYxo3tAzIOUCfMSHH8yH079RfhmEleiHmxrs"
access_token = "766306739738271745-Hn3HJm1FnJDEStdavMRBh3T7fzbLUWy"
access_token_secret = "G0YwIIM9FPOxchd0dEcyu0eoVVlZfgggxstDOwOuzLZbD"
authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)

import os
data_folder = os.path.join(os.path.expanduser("~"), "Data", "twitter")
output_filename = os.path.join(data_folder, "python_tweets.json")

import json
original_users = []
tweets = []
user_ids = {}

t = twitter.Twitter(auth=authorization)
search_results = t.search.tweets(q="python", count=100)['statuses']
for tweet in search_results:
    if 'text' in tweet:
        original_users.append(tweet['user']['screen_name'])
        user_ids[tweet['user']['screen_name']] = tweet['user']['id']
        tweets.append(tweet['text'])
print len(tweets)

from sklearn.externals import joblib
from sklearn.naive_bayes import BernoulliNB
model_filename = os.path.join(os.path.expanduser("~"), "Models", "twitter", "python_context.pkl")
joblib.dump(BernoulliNB,model_filename)

from sklearn.base import TransformerMixin
from nltk import word_tokenize
class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [{word: True for word in word_tokenize(document)}
                for document in X]

#调用joblib的load函数加载模型
from sklearn.externals import joblib
context_classifier = joblib.load(model_filename)

#调用context_classifier模型predict函数,预测消息是否与编程有关
y_pred = context_classifier.predict(tweets)
relevant_tweets = [tweets[i] for i in range(len(tweets)) if y_pred[i] == 1]
relevant_users = [original_users[i] for i in range(len(tweets)) if y_pred[i] == 1]
print(len(relevant_tweets))