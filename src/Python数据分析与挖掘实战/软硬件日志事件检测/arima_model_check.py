#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 16:43
# @Author  : Aries
# @Site    : 
# @File    : arima_model_check.py
# @Software: PyCharm
#模型检验
import pandas as pd

#参数初始化
discfile = u'拓展思考样本数据.xls'
lagnum = 12 #残差延迟个数

data = pd.read_excel(discfile, index_col = u'日期')
xdata = data[u'日志类告警']
print xdata

from statsmodels.tsa.arima_model import ARIMA #建立ARIMA(0,1,1)模型

arima = ARIMA(xdata.astype(float), (1, 1, 4)).fit() #建立并训练模型
xdata_pred = arima.predict(typ = 'levels') #预测
print xdata_pred
print arima.forecast(2)
pred_error = (xdata_pred - xdata).dropna() #计算残差

from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验

lb, p= acorr_ljungbox(pred_error, lags = lagnum)
h = (p < 0.05).sum() #p值小于0.05，认为是非白噪声。
if h > 0:
  print(u'模型ARIMA(0,1,1)不符合白噪声检验')
else:
  print(u'模型ARIMA(0,1,1)符合白噪声检验')