#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/25 14:41
# @Author  : Aries
# @Site    : 
# @File    : test.py
# @Software: PyCharm

from operator import itemgetter
import string
from sklearn.ensemble import BaggingRegressor
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn import linear_model
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import cross_validation
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report

# 输出得分
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#清理和处理数据
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

le = preprocessing.LabelEncoder()
enc=preprocessing.OneHotEncoder()

def clean_and_munge_data(df):
    #处理缺省值
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    #处理一下名字，生成Title字段
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    #处理特殊的称呼，全处理成mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme','Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms','Miss']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title =='':
            if x['Sex']=='Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title']=df.apply(replace_titles, axis=1)

    #看看家族是否够大，咳咳
    df['Family_Size']=df['SibSp']+df['Parch']
    df['Family']=df['SibSp']*df['Parch']


    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    df['AgeFill']=df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]

    df['AgeCat']=df['AgeFill']
    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'
    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'
    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'
    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')


    df.loc[ df.Cabin.isnull()==True,'Cabin'] = 0.5
    df.loc[ df.Cabin.isnull()==False,'Cabin'] = 1.5

    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

    #Age times class

    df['AgeClass']=df['AgeFill']*df['Pclass']
   # df['ClassFare']=df['Pclass']*df['Fare_Per_Person']


    df['HighLow']=df['Pclass']
    df.loc[ (df.Fare_Per_Person<8) ,'HighLow'] = 'Low'
    df.loc[ (df.Fare_Per_Person>=8) ,'HighLow'] = 'High'



    le.fit(df['Sex'] )
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(np.float)

    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title=le.transform(df['Title'])
    df['Title'] =x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl=le.transform(df['HighLow'])
    df['HighLow']=x_hl.astype(np.float)


    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb=le.transform(df['Embarked'])
    df['Embarked']=x_emb.astype(np.float)

    df = df.drop(['Name','Age','Cabin'], axis=1) #remove Name,Age and PassengerId


    return df


if __name__ == '__main__':
 #读取数据
 traindf=pd.read_csv("train.csv")
 ##清洗数据
 df=clean_and_munge_data(traindf)
 data_test = pd.read_csv("test.csv")
 df_test = clean_and_munge_data(data_test)

 train_df = df.filter(regex='Survived|Fare_Per_Person|Fare|AgeCat|Sex|Pclass|Family|Family_Size|Title')
 train_np = train_df.as_matrix()

 # y即Survival结果
 y = train_np[:, 0]

 # X即特征属性值
 X = train_np[:, 1:]

 # fit到BaggingRegressor之中
 clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
 bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
 bagging_clf.fit(X, y)

 test = df_test.filter(regex='Survived|Fare_Per_Person|Fare|AgeCat|Sex|Pclass|Family|Family_Size|Title')
 predictions = bagging_clf.predict(test)
 #result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
 #result.to_csv("logistic_regression_bagging_predictions3.csv", index=False)
 average = 0
 # 分割数据，按照 训练数据:cv数据 = 7:3的比例
 for i in range(0, 10):
  split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=5)
  train_df = split_train.filter(regex='Survived|Fare_Per_Person|Fare|AgeCat|Sex|Pclass|Family|Family_Size|Title')
  # 生成模型
  clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
  clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])


  # 对cross validation数据进行预测

  cv_df = split_cv.filter(regex='Survived|Fare_Per_Person|Fare|AgeCat|Sex|Pclass|Family|Family_Size|Title')
  predictions = clf.predict(cv_df.as_matrix()[:, 1:])
  y_test = cv_df.as_matrix()[:, 0]
  p = np.mean(predictions == y_test)
  print(p)
  average += p

  origin_data_train = pd.read_csv("train.csv")
  #print split_cv
  bad_cases = origin_data_train.loc[
     origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
  #print bad_cases






