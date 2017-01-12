#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 17:06
# @Author  : Aries
# @Site    : 
# @File    : analyzeImage.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import pandas as pd
car_data = 'car.xls'
data = pd.read_excel(car_data,index_col=u"纳税人编号") #读入数据
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

Survived_0 = data[data[u'输出']== 0][u'销售模式'].value_counts()
Survived_1 = data[data[u'输出']== 1][u'销售模式'].value_counts()
df=pd.DataFrame({u'正常':Survived_1, u'异常':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各销售模式的获救情况")
plt.xlabel(u"销售模式")
plt.ylabel(u"人数")
plt.show()