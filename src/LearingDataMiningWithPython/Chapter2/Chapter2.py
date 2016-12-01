#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/1 16:10
# @Author  : Aries
# @Site    : 
# @File    : Chapter2.py
# @Software: PyCharm

#导入numpy库和csv库
import numpy as np
import csv

data_filename = "ionosphere.data"

#创建Numpy数组xH和y存放的数据集
x=np.zeros((351,34),dtype=float)
y=np.zeros((351,),dtype=bool)

#用csv模式导入数据集
with open(data_filename,'r') as input_file:
   reader = csv.reader(input_file)
   for i,row in enumerate(reader):
       data = [float(datum) for datum in row[:-1]]
       x[i] = data
       y[i] = row[-1] == 'g'

#sciket-learning将数据库分为训练集和测试集的函数
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=14)

#初始化K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()
estimator.fit(X_train,y_train)

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_predicted == y_test) *100
#print accuracy
#print "The test accuracy is {:.1f}%".format(accuracy)

#交叉检验
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator,x,y,scoring='accuracy')
average_accuracy = np.mean(scores) *100
#print accuracy
#print "The test average_accuracy is {:.1f}%".format(average_accuracy)

#测试一下一系列n_neighbors的效果
avg_scorces = []
all_scorces = []
parameter_values = list(range(1,21))
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator,x,y,scoring='accuracy')
    avg_scorces.append(np.mean(scores))
    all_scorces.append(scores)

#用matplotlib来看图
from matplotlib import pyplot as plt
plt.plot(parameter_values,avg_scorces,'-o')
#plt.show()

#标准预处理
from sklearn.preprocessing import MinMaxScaler
X_transformed = MinMaxScaler().fit_transform(x)
estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator,X_transformed,y,scoring='accuracy')
#print "The test average_accuracy is {:.1f}%".format(np.mean(transformed_scores) * 100)

#创建流水线
from sklearn.pipeline import  Pipeline
scaling_pipeline = Pipeline([('scale',MinMaxScaler()),('predict',KNeighborsClassifier())])
scores = cross_val_score(scaling_pipeline, X_transformed, y, scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)
