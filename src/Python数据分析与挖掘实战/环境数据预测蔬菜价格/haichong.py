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
#print data.describe()

#选择我所需要的数据列
feature=[1,5,6,11,12]
y = [13]

y=data[y]
data = data[feature]

data_mean = data.mean()

data_std = data.std()

data = (data - data_mean)/data_std #数据标准化
data[13]= y

data = data.as_matrix()
shuffle(data) #随机打乱数据


def netBuild(ds):
    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（4个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(5, name='inLayer')
    hiddenLayer = SigmoidLayer(100, name='hiddenLayer0')
    outLayer = LinearLayer(5, name='outLayer')

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
    trainer = BackpropTrainer(fnn, ds, verbose=True, learningrate=0.04)
    # trainer.train()
    trainer.trainEpochs(epochs=20)
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
    DS = ClassificationDataSet(5,nb_classes=5)
    for ele in data:
        DS.appendLinked((ele[0],ele[1],ele[2],ele[3],ele[4]), (ele[5]))
    dsTrain,dsTest = DS.splitWithProportion(0.8)
    return dsTrain, dsTest

dsTrain_test,dsTest_test= classDsBuild(data)
dsTrain_test._convertToOneOfMany(bounds=[0, 1])
dsTest_test._convertToOneOfMany(bounds=[0, 1])

print dsTrain_test['target']

dsTrain,dsTest = dsBuild(data)
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

    for j in range(0,5):
        if origin[j] == 1:
            really.append(j)
            break
    for k in range(1,5):
        if prediction[k] > max:
            max = prediction[k]
            index = k
    pred.append(index)

    yuanma.append(dsTest_test['target'][i])
    calma.append(netModel.activate(dsTest_test['input'][i]))



print really
print pred
#print yuanma
#print calma





