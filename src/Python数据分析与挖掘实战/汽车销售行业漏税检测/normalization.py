#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 17:28
# @Author  : Aries
# @Site    : 
# @File    : normalization.py
# @Software: PyCharm
import pandas as pd
import numpy as np
datafile = 'car.xls' #参数初始化
data = pd.read_excel(datafile,index_col=u"纳税人编号") #读取数据
data=data/10**np.ceil(np.log10(data.abs().max()))#小数定标规范化

df = pd.DataFrame()
df[u'销售类型'] = data[u'销售类型']
df[u'销售模式'] = data[u'销售模式']
df[u'实际盈利'] = data[u'汽车销售平均毛利']-data[u'维修毛利']
print df

