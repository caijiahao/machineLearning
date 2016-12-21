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
print records[0]["tz"]

#统计最常出现的时区
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print time_zones[:10]

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
print counts['America/New_York']
print len(time_zones)

#计算出现前10的时间
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

print  top_counts(counts)

#使用collections.Counter类简单实现上述函数
from collections import Counter
counts = Counter(time_zones)
print counts.most_common(10)

#用panda对时区进行统计
