#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/26 16:22
# @Author  : Aries
# @Site    : 
# @File    : 4-6_principal_component_analyze.py
# @Software: PyCharm

#主成分分析 降维
import pandas as pd

#参数初始化
inputfile = 'principal_component.xls'
outputfile = 'dimention_reducted.xls' #降维后的数据

data = pd.read_excel(inputfile, header = None) #读入数据

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
print pca.components_ #返回模型的各个特征向量
print pca.explained_variance_ratio_ #返回各个成分各自的方差百分比

pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)
print low_d
pd.DataFrame(low_d).to_excel(outputfile)#保存结果
pca.inverse_transform(low_d) #必要时可以用inverse_transform函数复原数据
