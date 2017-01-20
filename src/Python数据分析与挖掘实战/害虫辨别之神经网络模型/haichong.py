#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/20 11:57
# @Author  : Aries
# @Site    : 
# @File    : haichong.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

from pybrain.structure import *
from pybrain.tools.shortcuts import *;
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle #导入随机函数shuffle，用来打算数据

import pandas as pd
import numpy as np
zaima ='zaima.xlsx'
data = pd.read_excel(zaima,header=None)



feature = ['0','1', '2', '3', '4', '5', '6','7','8','9','10','11'] #特征所在列
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean)/data_std #数据标准化
data = data.as_matrix()
shuffle(data) #随机打乱数据
print data

def netBuild(ds):
    net = buildNetwork(6, 12, 1)

    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（3个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(12, name='inLayer')
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
    trainer = BackpropTrainer(fnn, ds, verbose=True, learningrate=0.1)
    # trainer.train()
    trainer.trainEpochs(epochs=20)
    trainer.trainUntilConvergence(maxEpochs=100)
    print "Finish training"
    return fnn

def dsBuild(data):
    ds = SupervisedDataSet(12, 1)
    for ele in data:
        ds.addSample((ele[0],ele[1],ele[2],ele[3],ele[4],ele[5],ele[6],ele[7],ele[8],ele[9],ele[10],ele[11]), (ele[12]))
    dsTrain,dsTest = ds.splitWithProportion(0.8)
    return dsTrain,dsTest

dsTrain,dsTest = dsBuild(data)
print dsTrain
netModel = netBuild(dsTrain)
#pred=[]
sum = 0
for i in range(0,len(dsTest['input'])):
   # error = netModel.activate((data_test[i]))
   # print error
   # print error*data_std['y']+data_mean['y']
   print dsTest['target'][i]
   prediction = netModel.activate(
       (dsTest['input'][i][0], dsTest['input'][i][1], dsTest['input'][i][2], dsTest['input'][i][3],dsTest['input'][i][4],dsTest['input'][i][5],dsTest['input'][i][6],dsTest['input'][i][7],dsTest['input'][i][8],dsTest['input'][i][9],dsTest['input'][i][10],dsTest['input'][i][11]))
   print prediction

   if((prediction>0)and(dsTest['target'][i]>0) or (prediction<0)and(dsTest['target'][i]<0)):
       sum =sum+1

print sum/len(dsTest)*100










