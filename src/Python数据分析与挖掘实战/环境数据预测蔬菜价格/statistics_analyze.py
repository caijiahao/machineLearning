#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: statistics_analyze.py
@time: 2017/1/25 13:24
"""
import pandas as pd

inputfile = 'stationl.csv' #销量数据路径
outputfile = 'stationl.csv' #输出数据路径
data = pd.read_csv(inputfile,header=None) #读取数据
#data[1][data[1]==0] = None
#data[1][data[1]==-1] = None
#data[3][data[3]==-1] = None
#data[4][data[4]==-1] = None
#data[5][data[5]==-1] = None
#data[6][data[6]==-1] = None
#data[7][data[7]==-1] = None
#data[8][data[8]==-1] = None
#data[9][data[9]==-1] = None
#data[10][data[10]==-1] = None
#data[11][data[11]==-1] = None
#data[12][data[12]==-1] = None
#data[13][data[13]==-1] = None
#data[14][data[14]==-1] = None
#data[15][data[15]==-1] = None
#data[16][data[16]==-1] = None
#data[17][data[17]==-1] = None

data = data[(data[1] > 0)&(data[1] < 20000)] #过滤异常数据
data.to_csv(outputfile,header=None)
