#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: cal.py
@time: 2017/1/25 20:05
"""
import pandas as pd #导入数据分析库Pandas
import openpyxl

common='./November/'
inputfile = '2016-11-' #销量数据路径
outputfile = 'Nov.xlsx' #输出数据路径

data1 = []

for i in range(10,31):
    data = pd.read_excel(common+inputfile+str(i)+'.xlsx',header=None) #读入数据
    data = data.mean()
    data1.append([inputfile+str(i),data[1],data[2],data[3],data[4],data[5],data[6],data[7]])

data2 = pd.DataFrame(data1)
data2.to_excel(outputfile)





