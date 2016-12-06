#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/6 12:58
# @Author  : Aries
# @Site    : 
# @File    : Chapter5.py
# @Software: PyCharm

import pandas as pd
adult_filename = "adult.data"
adult = pd.read_csv(adult_filename, header=None, names=["Age", "Work-Class", "fnlwgt", "Education","Education-Num", "Marital-Status", "Occupation","Relationship", "Race", "Sex", "Capital-gain",
                                                        "Capital-loss", "Hours-per-week", "Native-Country","Earnings-Raw"])
#print adult[:5]
adult.dropna(how='all',inplace=True)
print adult.columns
print adult["Hours-per-week"].describe()

#用数据框的unique得到所有的工作情况
print adult["Work-Class"].unique()

adult["LongHours"] = adult["Hours-per-week"] > 40

#首先用numpy创建一个简单的矩阵
import numpy as np
x = np.arange(30).reshape((10,3))
x[:,1] = 1
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold()
xt = vt.fit_transform(x)
#print vt.variances_

X = adult[["Age", "Education-Num", "Capital-gain", "Capital-loss","Hours-per-week"]].values
y = (adult["Earnings-Raw"] == ' >50K').values

#使用SelectBest转换器类,用卡方函数打分，初始化转换器
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
transformer = SelectKBest(score_func=chi2, k=3)

#使用卡方检验分类较好的三个特征
Xt_chi2 = transformer.fit_transform(X,y)
print transformer.scores_

#使用皮尔逊相关系数检验
from scipy.stats import pearsonr
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:, column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer.fit_transform(X, y)
print(transformer.scores_)

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
clf = DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
print("Chi2 performance: {0:.3f}".format(scores_chi2.mean()))
print("Pearson performance: {0:.3f}".format(scores_pearson.mean()))

from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array

#创建继承自mixin的类
class MeanDiscrete(TransformerMixin):
 def fit(self, X, y=None):
        X = as_float_array(X)
        self.mean = np.mean(X, axis=0)
        return self
 def transform(self, X):
       X = as_float_array(X)
       assert X.shape[1] == self.mean.shape[0]
       return X > self.mean

mean_discrete = MeanDiscrete()
X_mean = mean_discrete.fit_transform(X)

import numpy as np
from numpy.testing import assert_array_equal
def test_meandiscrete():
    X_test = np.array([[0, 2],
                       [3, 5],
                       [6, 8],
                       [ 9, 11],
                       [12, 14],
                       [15, 17],
                       [18, 20],
                       [21, 23],
                       [24, 26],
                       [27, 29]])
    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))
    X_transformed = mean_discrete.transform(X_test)
    X_expected = np.array([[0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [1, 1],
                           [1, 1],
                           [1, 1],
                           [1, 1],
                           [1, 1]])
    assert_array_equal(X_transformed, X_expected)
test_meandiscrete()

#组装起来
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('mean_discrete', MeanDiscrete()),
                     ('classifier', DecisionTreeClassifier(random_state=14))])
scores_mean_discrete = cross_val_score(pipeline, X, y, scoring='accuracy')
print("Mean Discrete performance: {0:.3f}".format(scores_mean_discrete.mean()))
