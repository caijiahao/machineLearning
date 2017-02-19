#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/18 19:51
# @Author  : Aries
# @Site    : 
# @File    : stationarity.py
# @Software: PyCharm


#平稳性检验
import pandas as pd

#参数初始化
discfile = 'Octorber.xlsx'
predictnum =5 #不使用最后5个数据

data = pd.read_excel(discfile)
print data
data = data.iloc[: len(data)-5] #不检测最后5个数据

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
diff = 0
adf = ADF(data[8])
while adf[1] > 0.05:
  diff = diff + 1
  adf = ADF(data[8].diff(diff).dropna())

print(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(diff, adf[1]))