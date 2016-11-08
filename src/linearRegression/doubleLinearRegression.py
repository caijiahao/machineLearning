# -*- coding: gbk -*-  das as pd

import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
  
data = pd.read_csv('Advertising.csv')
print data

# visualize the relationship between the features and the response using scatterplots  
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.8,kind='reg')  
plt.show()

#create a python list of feature names  
feature_cols = ['TV', 'Radio', 'Newspaper']  
# use the list to select a subset of the original DataFrame  
X = data[feature_cols]  
# equivalent command to do this in one line  
X = data[['TV', 'Radio', 'Newspaper']]  
# print the first 5 rows  
print X.head()  
# check the type and shape of X  
print type(X)  
print X.shape  

# select a Series from the DataFrame  
y = data['Sales']  
# equivalent command that works if there are no spaces in the column name  
y = data.Sales  
# print the first 5 values  
print y.head()  

X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1) 
print X_train.shape  
print y_train.shape  
print X_test.shape  
print y_test.shape

linreg = LinearRegression()  
model=linreg.fit(X_train, y_train)  
print model  
print linreg.intercept_  
print linreg.coef_

# pair the feature names with the coefficients  
print zip(feature_cols, linreg.coef_)

y_pred = linreg.predict(X_test)  
print y_pred 
print type(y_pred) 

#计算Sales预测的RMSE  
print type(y_pred),type(y_test)  
print len(y_pred),len(y_test)  
print y_pred.shape,y_test.shape  
from sklearn import metrics  
import numpy as np  
sum_mean=0  
for i in range(len(y_pred)):  
    sum_mean+=(y_pred[i]-y_test[i])**2  
sum_erro=np.sqrt(sum_mean/50)  
# calculate RMSE by hand  
print "RMSE by hand:",sum_erro 

import matplotlib.pyplot as plt  
plt.figure()  
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")  
plt.plot(range(len(y_pred)),y_test,'r',label="test")  
plt.legend(loc="upper right") #显示图中的标签  
plt.xlabel("the number of sales")  
plt.ylabel('value of sales')  
plt.show()   