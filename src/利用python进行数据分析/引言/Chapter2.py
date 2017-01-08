#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: Chapter2.py
@time: 2016/12/20 22:33
"""

path = 'usagov_bitly_data2012-03-16-1331923249.txt'
print open(path).readline()

#json转换为字典对象
import json
records = [json.loads(line) for line in open(path)]
#print records[0]["tz"]

#统计最常出现的时区
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
#print time_zones[:10]

#计数
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

counts = get_counts(time_zones)
#print counts['America/New_York']
#print len(time_zones)

#计算出现前10的时间
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

#print  top_counts(counts)

#使用collections.Counter类简单实现上述函数
from collections import Counter
counts = Counter(time_zones)
#print counts.most_common(10)

#用panda对时区进行统计
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
frame = DataFrame(records)
#print frame

#print frame['tz'][:10]

tz_counts = frame['tz'].value_counts()
#print tz_counts[:10]

#替换缺失值和未知值
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
#print tz_counts[:10]

#利用counts对象的plot方法即可得到一张水平条形图
from pylab import *
tz_counts[:10].plot(kind='barh',rot=0)
#show()

#print frame['a'][1]
#print frame['a'][50]
#print frame['a'][51]

#将agent字符串分离出来并得到另外一份用户行为摘要
results = Series([x.split()[0] for x in frame.a.dropna()])
#print results[:5]

#print results.value_counts()[:8]

#统计window用户和非window用户的数量
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows', 'Not Windows')
#print operating_system[:5]

by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
#print agg_counts[:10]

indexer = agg_counts.sum(1).argsort()
#print indexer[:10]

count_subset = agg_counts.take(indexer)[-10:]
#print count_subset

count_subset.plot(kind='barh', stacked=True)
#show()

normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
#show()

#导入三个数据集
upath = 'users.dat'
rpath = 'ratings.dat'
mpath = 'movies.dat'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

users = pd.read_csv(upath, sep='::', header=None, names=unames)
ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames)
movies = pd.read_csv(mpath, sep='::', header=None, names=mnames)

#print users[:5]
#print ratings[:5]
#print movies[:5]
#print ratings

#合并三张表格
data = pd.merge(pd.merge(ratings,users),movies)
#print data
#print data.ix[0]

#按性别计算电影得分的平均值
mean_ratings = data.pivot_table('rating', index='title',columns='gender', aggfunc='mean')
#print mean_ratings[:5]

#赛选评论数大于250条的电影
ratings_by_title = data.groupby('title').size()
#print ratings_by_title[:5]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
#print active_titles[:10]
mean_ratings = mean_ratings.ix[active_titles]
#print mean_ratings

#为了了解女性观众最喜欢的电影，对F列降序排列
mean_ratings = mean_ratings.rename(index={'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)':'Seven Samurai (Shichinin no samurai) (1954)'})
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
#print top_female_ratings[:10]

#计算平均得分之差的列，并对其进行排序
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
#print sorted_by_diff[:15]

#对排序结果反序得到的就是男性更在喜欢的电影
print sorted_by_diff[::-1][:15]

#根据电影名称分组得分数据的标准差
rating_std_by_title = data.groupby('title')['rating'].std()
#根据active_titles进行过滤
rating_std_by_title = rating_std_by_title.ix[active_titles]
#根据值对Series进行降序排列
#print rating_std_by_title.order(ascending=False)[:10]

