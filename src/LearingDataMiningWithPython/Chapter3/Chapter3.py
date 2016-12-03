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
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.grid_search import GridSearchCV

data_filename = "nba.csv"
#数据集清洗
dataset = pd.read_csv(data_filename,parse_dates=["Date"])
dataset.columns = ["Date","Start(ET)","Visitor Team","VisitorPts","Home Team","HomePts","Score Type","OT?","Notes"]
#输出数据前五行
#print dataset.ix[:5]

#提取新特征
dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
dataset["HomeLastWin"] = 0
dataset["VisitorLastWin"] = 0
y_true = dataset["HomeWin"].values

#储存上次比赛的结果
from collections import defaultdict
won_last = defaultdict(int)
#print dataset.ix[:5]
for index,row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    dataset.ix[index,"HomeLastWin"] = row["HomeLastWin"]
    dataset.ix[index,"VisitorLastWin"] = row["VisitorLastWin"]
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]
#print dataset.ix[20:25]

#创建决策树
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)

x_previouswins = dataset[["HomeLastWin","VisitorLastWin"]].values
scores = cross_val_score(clf,x_previouswins,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

standings_filename = "leagues_NBA_2013_standings_expanded-standings.csv"
stadings = pd.read_csv(standings_filename)

dataset["HomeTeamRanksHigher"] = 0
for index,row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
        #比较排名更新特征值
    home_rank = stadings[stadings["Team"] == home_team]["Rk"].values[0]
    visitor_rank = stadings[stadings["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank>visitor_rank)
    dataset.ix[index,"HomeTeamRanksHigher"] = row["HomeTeamRanksHigher"]
#print dataset.ix[:5]

x_homehigher = dataset[["HomeLastWin","VisitorLastWin","HomeTeamRanksHigher"]].values
scores = cross_val_score(clf,x_homehigher,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

#创建新特征--上一次主场赢球
last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0
for index,row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team,visitor_team]))
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    dataset.ix[index] = row

    #更新last_match_winner字典
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner
x_lastwinner = dataset[["HomeTeamRanksHigher","HomeTeamWonLast"]].values
scores = cross_val_score(clf,x_lastwinner,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

#相关属性字符串转为整形
from sklearn.preprocessing import LabelEncoder
encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
x_teams = np.vstack([home_teams,visitor_teams]).T

#把整数转化成二进制数字
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
x_teams_expanded = onehot.fit_transform(x_teams).todense()
scores = cross_val_score(clf,x_teams_expanded,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

#随机森林解决问题
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
x_teams_expanded = onehot.fit_transform(x_teams).todense()
scores = cross_val_score(clf,x_teams_expanded,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

#随机森林采用不同的特征学习
x_all = np.hstack([x_homehigher,x_teams])
scores = cross_val_score(clf,x_all,y_true,scoring='accuracy')
print "The test average_accuracy is {:.1f}%".format(np.mean(scores) * 100)

#最佳参数设置
parameter_space = {
    "max_features":[1,5,'auto'],
    "n_estimators":[100,],
    "criterion":["gini","entropy"],
    "min_samples_leaf":[2,4,6]
}
grid = GridSearchCV(clf,parameter_space)
grid.fit(x_all,y_true)
print "The test average_accuracy is {:.1f}%".format(grid.best_score_ * 100)


