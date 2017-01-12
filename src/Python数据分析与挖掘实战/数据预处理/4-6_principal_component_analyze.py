#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/26 16:22
# @Author  : Aries
# @Site    : 
# @File    : 4-6_principal_component_analyze.py
# @Software: PyCharm

#主成分分析 降维
import pandas as pd

#参数初始化
inputfile = 'principal_component.xls'
outputfile = 'dimention_reducted.xls' #降维后的数据

data = pd.read_excel(inputfile, header = None) #读入数据


