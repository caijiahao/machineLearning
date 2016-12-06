#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/6 15:17
# @Author  : Aries
# @Site    : 
# @File    : Advertisements.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

data_filename = "ad.data"

#转换数据集中的字符串为数值类型
def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

from collections import defaultdict
converters = defaultdict(convert_number)
#将最后一列转换为0或者1
converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
converters[0] = lambda x:np.nan if x.strip() =="?" else float(x)
converters[1] = lambda x:np.nan if x.strip() =="?" else float(x)
converters[2] = lambda x:np.nan if x.strip() =="?" else float(x)
converters[3] = lambda x:np.nan if x.strip() =="?" else float(x)
ads = pd.read_csv(data_filename, header=None, converters=converters)
ads.dropna(inplace=True)
print ads[:-5]
X = ads.drop(1558, axis=1).values
y = ads[1558]

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)

np.set_printoptions(precision=3, suppress=True)
print pca.explained_variance_ratio_

clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
print("The average score from the reduced dataset is {:.4f}".format(np.mean(scores_reduced)))

from matplotlib import pyplot as plt
classes = set(y)
colors = ['red', 'green']
for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask, 0], Xd[mask, 1], marker='o', color=color, label=int(cur_class))
plt.legend()
plt.show()