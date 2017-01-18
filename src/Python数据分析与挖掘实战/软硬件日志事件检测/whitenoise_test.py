#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 15:53
# @Author  : Aries
# @Site    : 
# @File    : whitenoise_test.py
# @Software: PyCharm
#白噪声检验
import pandas as pd

#参数初始化
discfile = u'拓展思考样本数据.xls'

data = pd.read_excel(discfile)

#白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox

[[lb], [p]] = acorr_ljungbox(data[u'日志类告警'].diff().dropna(), lags = 1)
if p < 0.05:
  print(u'原始序列为非白噪声序列，对应的p值为：%s' %p)
else:
  print(u'原始该序列为白噪声序列，对应的p值为：%s' %p)

[[lb], [p]] = acorr_ljungbox(data[u'PING告警'].diff(1).dropna(), lags = 1)
if p < 0.05:
  print(u'一阶差分序列为非白噪声序列，对应的p值为：%s' %p)
else:
  print(u'一阶差分该序列为白噪声序列，对应的p值为：%s' %p)