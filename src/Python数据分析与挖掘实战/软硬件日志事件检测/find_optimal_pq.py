#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 16:07
# @Author  : Aries
# @Site    : 
# @File    : find_optimal_pq.py
# @Software: PyCharm

#确定最佳p、d、q值
import pandas as pd

#参数初始化
discfile = u'拓展思考样本数据.xls'

data = pd.read_excel(discfile, index_col =u'日期')
xdata = data[u'日志类告警'].astype(float)

from statsmodels.tsa.arima_model import ARIMA

#定阶
pmax = int(len(xdata)/10) #一般阶数不超过length/10
qmax = int(len(xdata)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(xdata, (p,0,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)


bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))