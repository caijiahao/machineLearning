#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 15:20
# @Author  : Aries
# @Site    : 
# @File    : car.py
# @Software: PyCharm

import pandas as pd #导入数据分析库Pandas

inputfile = 'think.xls' #输入数据路径,需要使用Excel格式；
outputfile = 'car.xls' #输出数据路径,需要使用Excel格式

data = pd.read_excel(inputfile,index_col=u"纳税人编号") #读入数据


#print data[data.notnull()]#查看缺失值情况
category = pd.Categorical(data[u'销售类型'])
data[u'销售类型'] = category.codes

category = pd.Categorical(data[u'销售模式'])
data[u'销售模式'] = category.codes

category = pd.Categorical(data[u'输出'])
data[u'输出'] = category.codes

data.to_excel(outputfile) #输出结果






