#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 17:28
# @Author  : Aries
# @Site    : 
# @File    : normalization.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import pandas as pd #导入数据分析库
from random import shuffle #导入随机函数shuffle，用来打算数据
import matplotlib.pyplot as plt
datafile = 'car.xls' #参数初始化
outputfile = 'dimention_reducted.xls' #降维后的数据
data = pd.read_excel(datafile,index_col=u"纳税人编号") #读取数据
data[u'实际盈利'] = data[u'汽车销售平均毛利']+data[u'维修毛利']

data=data/10**np.ceil(np.log10(data.abs().max()))#小数定标规范化

#print(data.corr()[u'输出'] ) #只显示“百合酱蒸凤爪”与其他菜式的相关系数

df = pd.DataFrame()
df[u'销售模式'] = data[u'销售模式']
df[u'实际盈利'] = data[u'实际盈利']
df[u'整体税负控制数'] = data[u'整体税负控制数']
df[u'代办保险率'] = data[u'代办保险率']
df[u'保费返还率'] = data[u'保费返还率']
df[u'输出'] = data[u'输出']

print df

data = df.as_matrix() #将表格转换为矩阵
shuffle(data) #随机打乱数据

p = 0.7 #设置训练数据比例
train = data[:int(len(data)*p),:] #前80%为训练集
test = data[int(len(data)*p):,:] #后20%为测试集

#构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型

treefile = 'tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(train[:,:5], train[:,5]) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(train[:,5], tree.predict(train[:,:5])).show() #显示混淆矩阵可视化结果
#注意到Scikit-Learn使用predict方法直接给出预测结果。


from sklearn.metrics import roc_curve #导入ROC曲线函数

fpr, tpr, thresholds = roc_curve(test[:,5], tree.predict_proba(test[:,:5])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果





