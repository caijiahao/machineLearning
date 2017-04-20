#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/21 15:19
# @Author  : Aries
# @Site    : 
# @File    : haichong.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import *;
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle #导入随机函数shuffle，用来打算数据

import pandas as pd
haichong='haichong-test.xlsx'
outputfile = 'result.xls'

data = pd.read_excel(haichong,header=None)
data_test = data
#print data.describe()

#选择我所需要的数据列
feature=[1,5,6,10,12]
y = [13]

y=data[y]
data = data[feature]

data_mean = data.mean()

data_std = data.std()

data = (data - data_mean)/data_std #数据标准化
data[13]= y

print data_test.corr()[13]

data = data.as_matrix()
shuffle(data) #随机打乱数据


def netBuild(ds):
    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（4个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(5, name='inLayer')
    hiddenLayer = SigmoidLayer(24, name='hiddenLayer0')
    #hiddenLayer1 = SigmoidLayer(4,name='hiddenLayer1')
   # hiddenLayer2 = SigmoidLayer(50, name='hiddenLayer2')
    outLayer = LinearLayer(4, name='outLayer')

    # 将五层都加入神经网络（即加入神经元）
    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer)
    #fnn.addModule(hiddenLayer1)
   # fnn.addModule(hiddenLayer2)
    fnn.addOutputModule(outLayer)

    # 建立五层之间的连接
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    #hidden_to_hidden = FullConnection(hiddenLayer,hiddenLayer1)
    #hidden_to_hidden1 = FullConnection(hiddenLayer1,outLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    # 将连接加入神经网络
    fnn.addConnection(in_to_hidden)
    #fnn.addConnection(hidden_to_hidden)
    #fnn.addConnection(hidden_to_hidden1)
    fnn.addConnection(hidden_to_out)

    # 让神经网络可用
    fnn.sortModules()

    print "Trainging"
    trainer = BackpropTrainer(fnn, ds, verbose=True, learningrate=0.01,momentum=0.95)
    # trainer.train()
    #trainer.trainEpochs(epochs=10)
    trainer.trainUntilConvergence(maxEpochs=10000)
    print "Finish training"
    return fnn

def dsBuild(data):
    ds = SupervisedDataSet(5, 1)
    for ele in data:
        ds.addSample((ele[0],ele[1],ele[2],ele[3],ele[4]), (ele[5]))
    dsTrain,dsTest = ds.splitWithProportion(0.9)
    return dsTrain,dsTest

def classDsBuild(data):
    DS = ClassificationDataSet(5,nb_classes=4)
    for ele in data:
        DS.appendLinked((ele[0],ele[1],ele[2],ele[3],ele[4]), (ele[5]))
    dsTrain,dsTest = DS.splitWithProportion(0.8)
    return dsTrain, dsTest


dsTrain_test,dsTest_test= classDsBuild(data)

#将类别转化为5位
dsTrain_test._convertToOneOfMany(bounds=[0, 1])
dsTest_test._convertToOneOfMany(bounds=[0, 1])

print dsTrain_test['target']

#划分训练集跟测试集
dsTrain,dsTest = dsBuild(data)
#训练神经网络
netModel = netBuild(dsTrain_test)
pred=[]
really =[]
yuanma = []
calma = []

for i in range(0,len(dsTest_test['input'])):

    origin = dsTest_test['target'][i]
    prediction = netModel.activate(dsTest_test['input'][i])
    max = prediction[0]
    index = 0

    for j in range(0,4):
        if origin[j] == 1:
            really.append(j)
            break
    for k in range(1,4):
        if prediction[k] > max:
            max = prediction[k]
            index = k
    pred.append(index)

    yuanma.append(dsTest_test['target'][i])
    calma.append(netModel.activate(dsTest_test['input'][i]))



print really
print pred

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






