#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: lagrange_newton_interp.py
@time: 2017/1/22 22:01
"""

import pandas as pd #导入数据分析库Pandas
from scipy.interpolate import lagrange #导入拉格朗日插值函数

inputfile = 'December.xlsx' #销量数据路径
outputfile = 'December_result.xlsx' #输出数据路径

data = pd.read_excel(inputfile) #读入数据

#计算相关性
data_test = data[[1,2,3,4,5,6,7,8,9,10,11,12,13]]
#data_test_mean = data.mean()
#data_test_std = data.std()
#data_test = (data_test-data_test_mean)/data_test_std

print data_test.corr()[13]

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
    y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果

#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i].isnull())[j]: #如果为空即插值。
      data[i][j] = ployinterp_column(data[i], j).round(0)

#data.to_csv(outputfile)
data.to_excel(outputfile) #输出结果，写入文件