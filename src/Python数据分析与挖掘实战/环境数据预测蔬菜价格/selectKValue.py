#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
haichong='haichong-test-origin.xlsx'

data = pd.read_excel(haichong, header=None)

col = 13
data = data[col]


#我们计算K值从1到10对应的平均畸变程度：
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist

from sklearn import metrics
K=range(2,10)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data.reshape((len(data),1)))
    meandistortions.append(sum(np.min(
            cdist(data.reshape((len(data),1)),kmeans.cluster_centers_,
                 'euclidean'),axis=1))/data.shape[0])
    print metrics.silhouette_score(data.reshape((len(data), 1)), kmeans.labels_, metric='euclidean')
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel(u'平均畸变程度')
plt.title(u'用肘部法则来确定最佳的K值')
plt.show()
