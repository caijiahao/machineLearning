#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 15:47
# @Author  : Aries
# @Site    : 
# @File    : stationarity_test.py
# @Software: PyCharm
##平稳性检验
import pandas as pd


#参数初始化
discfile = u'拓展思考样本数据.xls'

data = pd.read_excel(discfile)
#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
diff = 0
adf = ADF(data[u'日志类告警'].diff(2).dropna())
print adf[1]
while adf[1] > 0.05:
  diff = diff + 1
  adf = ADF(data[u'PING告警'].diff(diff).dropna())

print(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(diff, adf[1]))
