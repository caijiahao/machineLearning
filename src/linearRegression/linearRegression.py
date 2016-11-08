# Required Packages  
# -*- coding: utf-8 -*-  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
from sklearn import datasets, linear_model 

#Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name,sep=',', delimiter=',')
    print data
    X_parameter = []
    Y_parameter = []
    for single_square_feet,single_price_value in zip(data['time'],data['price']):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append([float(single_price_value)])
    return X_parameter,Y_parameter

def linear_model_main(X_parameters,Y_parameters,predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters,Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}  
    predictions['intercept'] = regr.intercept_  
    predictions['coefficient'] = regr.coef_  
    predictions['predicted_value'] = predict_outcome  
    return predictions  

X,Y = get_data('input_data.csv')  
predictvalue = 700  
result = linear_model_main(X,Y,predictvalue)  
print "Intercept value " , result['intercept']  
print "coefficient" , result['coefficient']  
print "Predicted value: ",result['predicted_value']