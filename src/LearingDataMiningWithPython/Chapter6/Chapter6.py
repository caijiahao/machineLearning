#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/8 21:26
# @Author  : Aries
# @Site    : 
# @File    : Chapter6.py
# @Software: PyCharm

#注意：因为是国外网站要用vpn
import IPython
import twitter

consumer_key = "GUK5q674gkJDcRh6syRatOcth"
consumer_secret = "LMBcY7nHkJfQWjZYxo3tAzIOUCfMSHH8yH079RfhmEleiHmxrs"
access_token = "766306739738271745-Hn3HJm1FnJDEStdavMRBh3T7fzbLUWy"
access_token_secret = "G0YwIIM9FPOxchd0dEcyu0eoVVlZfgggxstDOwOuzLZbD"
authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)

#保存搜索结果
import os
output_filename = "python_tweets.json"
labels_filename = "python_classes.json"
import json
#t = twitter.Twitter(auth=authorization)
#with open(output_filename, 'a') as output_file:
    #search_results = t.search.tweets(q="python", count=100)['statuses']
    #for tweet in search_results:
        #if 'text' in tweet:
            #print(tweet['user']['screen_name'])
            #print(tweet['text'])
            #print()
            #output_file.write(json.dumps(tweet))
            #output_file.write("\n\n")

tweets = []
with open(output_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line))
print("Loaded {} tweets".format(len(tweets)))

labels = []
tweet_sample = tweets
if os.path.exists(labels_filename):
    with open(labels_filename) as inf:
        labels = json.load(inf)

def get_next_tweet():
    return tweet_sample[len(labels)]['text']

#创建JavaScript来收集输入

