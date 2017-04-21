#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: svm.py
@time: 2017/3/4 23:03
"""

from sklearn.svm import SVC
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = SVC()
clf.fit(X, y)
print clf.fit(X, y)
print clf.predict([[-0.8, -1]])