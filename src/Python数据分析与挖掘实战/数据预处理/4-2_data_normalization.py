#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/26 15:28
# @Author  : Aries
# @Site    : 
# @File    : 4-2_data_normalization.py
# @Software: PyCharm

#数据规范化
import pandas as pd
import numpy as np

datafile = 'normalization_data.xls' #参数初始化
data = pd.read_excel(datafile, header = None) #读取数据

print (data - data.min())/(data.max() - data.min()) #最小-最大规范化
print (data - data.mean())/data.std() #零-均值规范化
print data/10**np.ceil(np.log10(data.abs().max())) #小数定标规范化