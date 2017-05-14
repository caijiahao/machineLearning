#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: Random.py
@time: 2017/4/26 15:52
"""

# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree import export_graphviz

import pandas as pd
haichong='haichong-test.xlsx'
outputfile = 'result.xls'

data = pd.read_excel(haichong,header=None)
data_test = data
#print data.describe()

#选择我所需要的数据列
feature=[5,9,10,12]
y = [13]

target=data[y].as_matrix()

feature = data[feature]

data_mean = feature.mean()

data_std = feature.std()

feature = (feature - data_mean)/data_std #数据标准化

feature = feature.as_matrix()


#拆分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.7,random_state=42)

#分类型决策树
clf = RandomForestClassifier(n_estimators = 10)

#训练模型
s = clf.fit(feature_train , target_train)
print s

#评估模型准确率
r = clf.score(feature_test , target_test)
print '评估模型的准确率',r


print '判定结果：%s' % clf.predict(feature_test[8])
print '原结果：%s' %target_test[8]
#print clf.predict_proba(feature_test[0])

print '所有的树:%s' % clf.estimators_

print clf.classes_
print clf.n_classes_

print '各feature的重要性：%s' % clf.feature_importances_

print clf.n_outputs_

def _parallel_helper(obj, methodname, *args, **kwargs):
    return getattr(obj, methodname)(*args, **kwargs)

all_proba = Parallel(n_jobs=10, verbose=clf.verbose, backend="threading")(
            delayed(_parallel_helper)(e, 'predict_proba', feature_test[8]) for e in clf.estimators_)
print '所有树的判定结果：%s' % all_proba

proba = all_proba[0]
for j in range(1, len(all_proba)):
    proba += all_proba[j]
proba /= len(clf.estimators_)
print '数的棵树：%s ， 判不作弊的树比例：%s' % (clf.n_estimators , proba[0,0])
print '数的棵树：%s ， 判作弊的树比例：%s' % (clf.n_estimators , proba[0,1])

#当判作弊的树多余不判作弊的树时，最终结果是判作弊
print '判断结果：%s' % clf.classes_.take(np.argmax(proba, axis=1), axis=0)
print '原结果：%s' %target_test[8]

#把所有的树都保存到word
for i in xrange(len(clf.estimators_)):
    export_graphviz(clf.estimators_[i] , '%d.dot'%i)