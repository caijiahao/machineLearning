#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 15:36
# @Author  : Aries
# @Site    : 
# @File    : correlation_analyze.py
# @Software: PyCharm

import pandas as pd

inputfile =u'拓展思考样本数据.xls'

data = pd.read_excel(inputfile,index_col=u'日期')
print data.corr()[u'故障类别']