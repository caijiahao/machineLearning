#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/4 18:24
# @Author  : Aries
# @Site    : 
# @File    : Chapter4.py
# @Software: PyCharm

#用pandas加载数据
import pandas as pd
import sys

ratings_filename = "ratings.csv"
all_ratings = pd.read_csv(ratings_filename,delimiter=",",header=None,names=["UserID","MovieID","Rating","Timestamp"])
all_ratings["Timestamp"] = pd.to_datetime(all_ratings['Timestamp'],unit = 's')
#print all_ratings[:5]

#增加一个特征看看该用户是否喜欢这部电影
all_ratings["Favorable"] = all_ratings["Rating"] > 3
#print all_ratings[10:15]
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
favorable_ratings = ratings[ratings["Favorable"]]

#按照UserID进行分组,并遍历每个用户看过的每一部电影
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("UserID")["MovieID"])
#print favorable_reviews_by_users

#统计每部电影被喜欢的程度
num_favorable_by_movie = ratings[["MovieID","Favorable"]].groupby("MovieID").sum()
#print num_favorable_by_movie.sort("Favorable",ascending=False)[:5]

#Ariori算法第一步
frequent_itemsets = {}
min_support = 50
frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"]) for movie_id,row in num_favorable_by_movie.iterrows() if row["Favorable"] > min_support)

from collections import defaultdict
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
          counts = defaultdict(int)
          for user, reviews in favorable_reviews_by_users.items():
              for itemset in k_1_itemsets:
                  if itemset.issubset(reviews):
                      for other_reviewed_movie in reviews - itemset:
                          current_superset = itemset | frozenset((other_reviewed_movie,))
                          counts[current_superset] += 1
          return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])


#循环调用Aprior算法
for k in range(2,20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users,frequent_itemsets[k-1],min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:
        print "Did not find any frequent itemsets of length {}".format(k)
        break;
    else:
        print "I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k)

#删除只有一个的频繁集
del frequent_itemsets[1]
print frequent_itemsets[10]







