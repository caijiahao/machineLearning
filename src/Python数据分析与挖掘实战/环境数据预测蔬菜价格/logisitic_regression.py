#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/21 16:03
# @Author  : Aries
# @Site    : 
# @File    : logisitic_regression.py
# @Software: PyCharm
#逻辑回归 自动建模
import pandas as pd

#参数初始化
filename = 'haichong-test.xlsx'
data = pd.read_excel(filename,header=None)
feature = [1,2,3,4,5,6,7,8,9,10,11,12]
label = [13]

x = data[feature].as_matrix()
y = data[label].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.cross_validation import cross_val_score

rlr = RLR() #建立随机逻辑回归模型，筛选变量
rlr.fit(x, y) #训练模型
print rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数

#print(u'通过随机逻辑回归模型筛选特征结束。')
#print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
#x = data[data.columns[rlr.get_support()]].as_matrix() #筛选好特征

lr = LR() #建立逻辑货柜模型
lr.fit(x, y) #用筛选后的特征数据来训练模型
print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s' % lr.score(x, y)) #给出模型的平均正确率，本例为81.4%
scores = cross_val_score(lr,x,y,cv=5)

#print y
#print lr.predict(x)
