#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/18 20:26
# @Author  : Aries
# @Site    : 
# @File    : huise.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from GM11 import GM11 #引入自己编写的灰色预测函数

inputfile = 'station5.csv' #输入的数据文件
outputfile = 'station5_data1_GM11.xls' #灰色预测后保存的路径
data = pd.read_csv(inputfile) #读取数据

l = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6','sensor7']
for i in l:
  f = GM11(data[i][range(0, len(data)-2)].as_matrix())[0]
  print f(2370)
  print f(2371)
  data[i][len(data)-2] = f(len(data)-2) #2014年预测结果
  data[i][len(data)-1] = f(len(data)-1) #2015年预测结果
  data[i] = data[i].round(1) #保留两位小数

data.to_excel(outputfile) #结果输出