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

common='./1/'
inputfile = '2017-01-' #销量数据路径
outputfile = 'Feb.xlsx' #输出数据路径

data1 = []

for i in range(10,26):
    data = pd.read_excel(common+inputfile+str(i)+'.xlsx',header=None) #读入数据

    #获取降雨量
    jiangyu = data[1]
    sum = 0
    for j in jiangyu:
        sum +=j
    data = data.mean()


    data1.append([inputfile+str(i),sum.round(1),data[2].round(1),data[3].round(1),data[4].round(1),data[5].round(1),data[6].round(1),data[7].round(1)])

data2 = pd.DataFrame(data1)
data2.to_excel(outputfile)





