#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/13 15:08
# @Author  : Aries
# @Site    : 
# @File    : 7-3_zscore_data.py
# @Software: PyCharm

#标准差标准化

import pandas as pd

datafile = 'zscoredata.xls' #需要进行标准化的数据文件；
zscoredfile = 'zscoreddata.xls' #标准差化后的数据存储路径文件；

#标准化处理
data = pd.read_excel(datafile)
data = (data - data.mean(axis = 0))/(data.std(axis = 0)) #简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。
data.columns=['Z'+i for i in data.columns] #表头重命名。

data.to_excel(zscoredfile, index = False) #数据写入