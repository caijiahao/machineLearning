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

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(test[:,6], tree.predict(test[:,:6])).show() #显示混淆矩阵可视化结果
#注意到Scikit-Learn使用predict方法直接给出预测结果。


from sklearn.metrics import roc_curve #导入ROC曲线函数

fpr, tpr, thresholds = roc_curve(test[:,6], tree.predict_proba(test[:,:6])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果