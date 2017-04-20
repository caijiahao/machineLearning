#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/5 18:01
# @Author  : Aries
# @Site    : 
# @File    : data_discretization.py
# @Software: PyCharm

if __name__ == '__main__':
 import pandas as pd
 haichong='haichong-test-origin.xlsx'
 outputfile = 'result.xls'

 data = pd.read_excel(haichong,header=None)

 col = 12
 test = data
 data = data[col].copy()

 k = 4

 from sklearn.cluster import KMeans
 kmodel = KMeans(n_clusters=k,n_jobs=4)
 kmodel.fit(data.reshape((len(data),1)))
 c = pd.DataFrame(kmodel.cluster_centers_).sort(0)

 w = pd.rolling_mean(c,2).iloc[1:]
 w = [0] + list(w[0]) + [data.max()]
 d3 = pd.cut(data,w,labels=range(k))
 print d3
 test[col] = d3
 #print test

 def cluster_plot(d,k):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8,3))
    for j in range(0,k):
        plt.plot(data[d==j],[j for i in d[d == j]],'o')

    plt.ylim(-0.5,k-0.5)
    return plt

 cluster_plot(d3,k).show()

 test.to_excel(outputfile,header=None)


