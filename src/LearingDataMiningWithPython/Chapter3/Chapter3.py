#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: Chapter3.py
@time: 2016/12/1 23:59
"""
#用read_csv函数就能加载数据集
import pandas as pd
data_filename = "nba.csv"
#数据集清洗
dataset = pd.read_csv(data_filename,parse_dates=["Date"])
dataset.columns = ["Date","Start(ET)","Visitor Team","VisitorPts","Home Team","HomePts","Score Type","OT?","Notes"]
#输出数据前五行
print dataset.ix[:5]
