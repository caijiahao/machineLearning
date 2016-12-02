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
#print dataset.ix[:5]

#提取新特征
dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["HomeWin"].values

#储存上次比赛的结果
from collections import defaultdict
won_last = defaultdict(int)
print dataset.ix[:5]
for index,row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    dataset.ix[index] = row
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]
#print dataset.ix[20:25]


