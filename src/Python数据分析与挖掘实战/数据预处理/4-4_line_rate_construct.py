#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/26 15:57
# @Author  : Aries
# @Site    : 
# @File    : 4-4_line_rate_construct.py
# @Software: PyCharm

#线损率属性构造
import pandas as pd

#参数初始化
inputfile= 'electricity_data.xls' #供入供出电量数据
outputfile = 'electricity_data.xls' #属性构造后数据文件

data = pd.read_excel(inputfile) #读入数据
data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量'])/data[u'供入电量']

data.to_excel(outputfile, index = False) #保存结果