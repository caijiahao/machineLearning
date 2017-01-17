#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/17 15:56
# @Author  : Aries
# @Site    : 
# @File    : surronding.py
# @Software: PyCharm

import pandas as pd
from random import shuffle #导入随机函数shuffle，用来打算数据
import matplotlib.pyplot as plt

inputfile = 'surronding.xls' #数据文件
outputfile1 = 'cm_train.xls' #训练样本混淆矩阵保存路径
outputfile2 = 'cm_test.xls' #测试样本混淆矩阵保存路径
data = pd.read_excel(inputfile) #读取数据，指定编码为gbk

#data[u'空气等级'][data[u'空气等级'] == u'I'] = 1
#data[u'空气等级'][data[u'空气等级'] == u'II'] = 2
#data[u'空气等级'][data[u'空气等级'] == u'III'] = 3
#data[u'空气等级'][data[u'空气等级'] == u'IV'] = 4
#data[u'空气等级'][data[u'空气等级'] == u'V'] = 5
#data[u'空气等级'][data[u'空气等级'] == u'VI'] = 6
data = data.as_matrix()

shuffle(data) #随机打乱数据

p = 0.8 #设置训练数据比例
train = data[:int(len(data)*p),:] #前80%为训练集
test = data[int(len(data)*p):,:] #后20%为测试集

#构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型

treefile = 'tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(train[:,:6], train[:,6]) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile)

#导入输出相关的库，生成混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(train[:,6], tree.predict(train[:,:6])) #训练样本的混淆矩阵
cm_test = metrics.confusion_matrix(test[:,6], tree.predict(test[:,:6])) #测试样本的混淆矩阵

#保存结果
pd.DataFrame(cm_train, index = range(1, 7), columns = range(1, 7)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 7), columns = range(1, 7)).to_excel(outputfile2)

