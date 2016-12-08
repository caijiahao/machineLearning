#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/8 21:26
# @Author  : Aries
# @Site    : 
# @File    : Chapter6.py
# @Software: PyCharm

import twitter
consumer_key = "52Nu7ubm2szT1JyJEOB7V2lGM"
consumer_secret = "mqA94defqjioyWeMxdJsSduthxdMMGd2vfOUKvOFpm0n7JTqfY"
access_token = "16065520-USf3DBbQAh6ZA8CnSAi6NAUlkorXdppRXpC4cQCKk"
access_token_secret = "DowMQeXqh5ZsGvZGrmUmkI0iCmI34ShFzKF3iOdiilpX5"
authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)

#保存搜索结果
import os
output_filename = "python_tweets.json"

import json
t = twitter.Twitter(auth=authorization)
with open(output_filename, 'a') as output_file:
    search_results = t.search.tweets(q="python", count=100)['statuses']
    for tweet in search_results:
        if 'text' in tweet:
            print(tweet['user']['screen_name'])
            print(tweet['text'])
            print()
            output_file.write(json.dumps(tweet))
            output_file.write("\n\n")

