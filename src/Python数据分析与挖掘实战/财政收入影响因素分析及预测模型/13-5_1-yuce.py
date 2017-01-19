#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pybrain.structure import *
from pybrain.tools.shortcuts import *;
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer



import pandas as pd
inputfile = 'data1_GM11.xls' #灰色预测后保存的路径
outputfile = 'revenue.xls' #神经网络预测后保存的结果
modelfile = '1-net.model' #模型保存路径
data = pd.read_excel(inputfile) #读取数据
feature = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7'] #特征所在列



data_train = data.loc[range(1994,2014)].copy() #取2014年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化
data_train = np.array(data_train)
data_test = data.loc[range(2014,2016)].copy() #取2014年后的数据建模
data_test = np.array((data_test[feature] - data_mean)/data_std)
print data_test






def netBuild(ds):
    net = buildNetwork(6, 12, 1)

    # 建立神经网络fnn
    fnn = FeedForwardNetwork()

    # 设立三层，一层输入层（3个神经元，别名为inLayer），一层隐藏层，一层输出层
    inLayer = LinearLayer(6, name='inLayer')
    hiddenLayer = SigmoidLayer(12, name='hiddenLayer0')
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
    trainer.trainUntilConvergence(maxEpochs=10000)
    print "Finish training"
    return fnn

def dsBuild(data):
    ds = SupervisedDataSet(6, 1)
    for ele in data:
        ds.addSample((ele[0],ele[1],ele[2],ele[3],ele[4],ele[5]), (ele[6]))
    dsTrain,dsTest = ds.splitWithProportion(0.8)
    return dsTrain,dsTest

dsTrain,dsTest = dsBuild(data_train)
netModel = netBuild(dsTrain)
dsTest = dsTrain
#pred=[]
for i in range(0,len(dsTest['input'])):
   # error = netModel.activate((data_test[i]))
   # print error
   # print error*data_std['y']+data_mean['y']
   print dsTest['target'][i]*data_std['y']+data_mean['y']
   prediction = netModel.activate(
       (dsTest['input'][i][0], dsTest['input'][i][1], dsTest['input'][i][2], dsTest['input'][i][3],dsTest['input'][i][4],dsTest['input'][i][5]))*data_std['y']+data_mean['y']
   print prediction




