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
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle #导入随机函数shuffle，用来打算数据

import pandas as pd
haichong='haichong-test.xlsx'
outputfile = 'result.xls'

data = pd.read_excel(haichong,header=None)
print data.describe()

#选择我所需要的数据列
feature=[1,5,6,11,12,13]
data = data[feature]

data_mean = data.mean()
print data_mean
data_std = data.std()
print data_std
data = (data - data_mean)/data_std #数据标准化
data = data.as_matrix()
shuffle(data) #随机打乱数据


def netBuild(ds):
    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（4个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(5, name='inLayer')
    hiddenLayer = SigmoidLayer(100, name='hiddenLayer0')
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
    trainer.trainUntilConvergence(maxEpochs=10000)
    print "Finish training"
    return fnn

def dsBuild(data):
    ds = SupervisedDataSet(5, 1)
    for ele in data:
        ds.addSample((ele[0],ele[1],ele[2],ele[3],ele[4]), (ele[5]))
    dsTrain,dsTest = ds.splitWithProportion(0.9)
    return dsTrain,dsTest

dsTrain,dsTest = dsBuild(data)
netModel = netBuild(dsTrain)
pred=[]
really =[]

for i in range(0,len(dsTest['input'])):
    really.append((dsTest['target'][i]*94+84).round(0))
    pred.append((netModel.activate(
       (dsTest['input'][i][0], dsTest['input'][i][1],dsTest['input'][i][2],dsTest['input'][i][3],dsTest['input'][i][4]))*94+84).round(0))


print really
print pred





