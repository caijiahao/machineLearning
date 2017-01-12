#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 16:54
# @Author  : Aries
# @Site    : 
# @File    : statistics_analyze.py
# @Software: PyCharm

import pandas as pd
car_data = 'car.xls'
data = pd.read_excel(car_data,index_col=u"纳税人编号") #读入数据
statistics = data.describe() #保存基本统计量

statistics.loc['range'] = statistics.loc['max']-statistics.loc['min'] #极差
statistics.loc['var'] = statistics.loc['std']/statistics.loc['mean'] #变异系数
statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%'] #四分位数间距

print(statistics)