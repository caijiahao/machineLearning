#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/28 13:02
# @Author  : Aries
# @Site    : 
# @File    : Chapter1.py
# @Software: PyCharm

import numpy as np
#加载读取数据
dataset_filename = "affinity_dataset.txt"
x = np.loadtxt(dataset_filename)
features = ['kaochang','milk','cheese','apple','bannana']
#print x[:5]

#计算支持度和置信度的方法
num_apple_purchases = 0
num_banner_purchases = 0
for sample in x:
    if sample[3] == 1:
        num_apple_purchases +=1
    if sample[4] == 1:
        num_banner_purchases += 1
print "{0} people bought Apples".format(num_apple_purchases)
print "{0} people bought Banner".format(num_banner_purchases)

#利用字典记录关联关系
from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)
for sample in x:
    for premise in range(4):
        if sample[premise] == 0:continue
        num_occurances[premise] += 1
        for conclusion in range(4):
            if premise == conclusion:continue
        if sample[conclusion] == 1:
            valid_rules[(premise,conclusion)]+=1
        else:
            invalid_rules[(premise,conclusion)]+=1
support = valid_rules
confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    rule = (premise,conclusion)
    confidence[rule] = valid_rules[rule] *1.0/ num_occurances[premise]
#print confidence
def print_rule(premise,conclusion,support,confidence,features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print "Rule:If a person buys {0} they will also buy {1}".format(premise_name,conclusion_name)
    print "- Support: {0}".format(support[premise,conclusion])
    print "- Confidence:{0:.3f}".format(confidence[(premise,conclusion)])

premise = 1
conclusion = 3
print_rule(premise,conclusion,support,confidence,features)

#列出支持度或者置信度前五的关联关系
from operator import itemgetter
sorted_support =sorted(support.items(),key=itemgetter(1),reverse=True)
for index in range(4):
    print "Rule #{0}".format(index+1)
    premise,conclusion = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)












