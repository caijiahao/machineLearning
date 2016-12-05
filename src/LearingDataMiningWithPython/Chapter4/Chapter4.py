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

ratings_filename = "u.data"
all_ratings = pd.read_csv(ratings_filename,delimiter="\t",header=None,names=["UserID","MovieID","Rating","Timestamp"])
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
#print frequent_itemsets[10]

#制造备选规则
candidate_rules = []
for itemset_length,itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise,conclusion))
#print candidate_rules[:5]

#计算每条规则的置信度
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}

#对置信字典进行排序，输出置信度最高的前五条规则
from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print "Rule #{0}".format(index + 1)
    (premise, conclusion) = sorted_confidence[index][0]
    print "Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion)
    print " - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)])
    print " "

#电影名称文件名
movie_name_filename = "u.item"
movie_name_data  = pd.read_csv(movie_name_filename,delimiter="|",encoding="mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action", "Adventure",
                           "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
#print movie_name_data[:5]

#创建找到电影名称的函数
def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
    title = title_object.values[0]
    return title

#加上movie正式名称后的输出
for index in range(4,9):
    print "Rule #{0}".format(index + 1)
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print "Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name)
    print " - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)])
    print " "

#评估,训练前200名的用户的打分数据,其他作为测试库
test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
test_favorable = test_dataset[test_dataset["Favorable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("UserID")["MovieID"])

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

test_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule]) for candidate_rule in rule_confidence}
print len(test_confidence)

sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(4,9):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print "Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name)
    print " - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1))
    print " - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1))
    print(" ")