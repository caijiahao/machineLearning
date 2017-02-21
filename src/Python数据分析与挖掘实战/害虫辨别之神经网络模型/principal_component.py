#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/19 20:25
# @Author  : Aries
# @Site    : 
# @File    : principal_component.py
# @Software: PyCharm

#主成分分析 降维
import pandas as pd

inputfile = 'train1.xlsx'
outputfile= 'dimention_reducted.xls'

data = pd.read_excel(inputfile, header = None) #读入数据
print data

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
#print pca.components_
#print pca.explained_variance_ratio_

pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)
#print low_d\
print pd.DataFrame(low_d)
#print pd.DataFrame(pca.inverse_transform(low_d))
