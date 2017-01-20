#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/20 11:57
# @Author  : Aries
# @Site    : 
# @File    : haichong.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import *;
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle #导入随机函数shuffle，用来打算数据

import pandas as pd
zaima ='zaima.xlsx'
data = pd.read_excel(zaima,header=None)
print data
print data.corr()[12]



feature = [1,3,4, 5,6,7,8,9,10,11,12] #特征所在列
data = data[feature]
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean)/data_std #数据标准化
data = data.as_matrix()
shuffle(data) #随机打乱数据
print data

def netBuild(ds):
    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（3个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(10, name='inLayer')
    hiddenLayer = SigmoidLayer(24, name='hiddenLayer0')
    outLayer = LinearLayer(1, name='outLayer')

    # 将三层都加入神经网络（即加入神经元）
    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer)
    fnn.addOutputModule(outLayer)

    # 建立三层之间的连接
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    # 将连接加入神经网络
    fnn.addConnection(in_to_hidden)
    fnn.addConnection(hidden_to_out)

    # 让神经网络可用
    fnn.sortModules()

    print "Trainging"
    trainer = BackpropTrainer(fnn, ds, verbose=True, learningrate=0.01)
    # trainer.train()
    trainer.trainEpochs(epochs=20)
    trainer.trainUntilConvergence(maxEpochs=1000)
    print "Finish training"
    return fnn

def dsBuild(data):
    ds = SupervisedDataSet(10, 1)
    for ele in data:
        ds.addSample((ele[0],ele[1],ele[2],ele[3],ele[4],ele[5],ele[6],ele[7],ele[8],ele[9]), (ele[10]))
    dsTrain,dsTest = ds.splitWithProportion(0.8)
    return dsTrain,dsTest

dsTrain,dsTest = dsBuild(data)
netModel = netBuild(dsTrain)
pred=[]
really =[]
sum = 0
for i in range(0,len(dsTest['input'])):
   print dsTest['target'][i]
   if(dsTest['target'][i]>0):
       really.append(1)
   else:
       really.append(0)
   prediction = netModel.activate(
       (dsTest['input'][i][0], dsTest['input'][i][1], dsTest['input'][i][2], dsTest['input'][i][3],dsTest['input'][i][4],dsTest['input'][i][5],dsTest['input'][i][6],dsTest['input'][i][7],dsTest['input'][i][8],dsTest['input'][i][9]))
   print prediction

   if(prediction>0):
       pred.append(1)
   else :
       pred.append(0)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(really, pred).show() #显示混淆矩阵可视化结果
#注意到Scikit-Learn使用predict方法直接给出预测结果。


from sklearn.metrics import roc_curve #导入ROC曲线函数
fpr, tpr, thresholds = roc_curve(really, pred, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of BP', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果










