#-*- coding: utf-8 -*-
import pandas as pd
inputfile = 'data1.csv' #输入的数据文件
data = pd.read_csv(inputfile) #读取数据

#导入AdaptiveLasso算法，要在较新的Scikit-Learn才有。
from sklearn.linear_model import Lasso
model = Lasso(alpha=1)
model.fit(data.iloc[:,0:13],data['y'])
print model.coef_ #各个特征的系数
