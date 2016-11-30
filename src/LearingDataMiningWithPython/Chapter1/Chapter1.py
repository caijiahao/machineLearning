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

#导入Iris植物数据集
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y =dataset.target
#print dataset.DESCR

#将数值型数据转为离散化
attribute_means = x.mean(axis=0)
x_d = np.array(x>=attribute_means,dtype='int')

#根据待测数据的某项特征值预测类别并给出错误率
from collections import defaultdict
from operator import itemgetter

def train_feature_value(x,y_true,feature_index,value):
    class_counts = defaultdict(int)
    for sample,y in zip(x,y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(),key=itemgetter(1),reverse=True)
    most_frequent_class = sorted_class_counts[0][0]

    #计算该条规则的错误率
    incorrect_predictions = [class_count for class_value,class_count
    in class_counts.items()
    if class_value  != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class,error

#找出每个特征值最可能的类别，并计算错误率
def train_on_feature(x,y_true,feature_index):
    values = set(x[:,feature_index])
    predictors = {}
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(x,y_true,feature_index,current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors,total_error

#sciket-learning将数据库分为训练集和测试集的函数
from sklearn.cross_validation import train_test_split
Xd_train,Xd_test,y_train,y_test = train_test_split(x_d,y,random_state=None)

#找出错误率最低的特征，作为分类的唯一规则
all_predictors = {}
errors = {}
for feature_index in range(Xd_train.shape[1]):
    predictors,total_error = train_on_feature(Xd_train,y_train,feature_index)
    all_predictors[feature_index] = predictors
    errors[feature_index] = total_error
    best_feature,best_error = sorted(errors.items(),key=itemgetter(1))[0]

#对预测器进行排序，找到最佳特征值，创建model模型
model = {'feature':best_feature,'predictor': all_predictors[best_feature]}

def predict(X_test,model):
    print model
    variable = model['feature']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(Xd_test,model)
accuracy = np.mean(y_predicted == y_test) *100
#print accuracy
print "The test accuracy is {:.1f}%".format(accuracy)




















