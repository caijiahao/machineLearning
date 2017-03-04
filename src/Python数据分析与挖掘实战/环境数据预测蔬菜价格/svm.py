#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: svm.py
@time: 2017/3/4 23:03
"""

import pandas as pd
from random import shuffle #导入随机函数shuffle，用来打算数据
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC


haichong='haichong-test.xlsx'

data = pd.read_excel(haichong,header=None)
#选择我所需要的数据列
feature=[1,5,6,11,12]
y = [13]

y=data[y]
data = data[feature]

data_mean = data.mean()

data_std = data.std()

data = (data - data_mean)/data_std #数据标准化
#data[13]= y

data = data.as_matrix()
shuffle(data) #随机打乱数据

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

clf = SVC(degree=5) #ovo为一对一
clf.fit(x_train,y_train)

really = []
predition = []
for i in range(0,16):
    really.append(y_test[i])
    predition.append(clf.predict(x_test[i]))

print really
print predition