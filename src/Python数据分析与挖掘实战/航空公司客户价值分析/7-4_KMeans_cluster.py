#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/13 15:28
# @Author  : Aries
# @Site    : 
# @File    : 7-4_KMeans_cluster.py
# @Software: PyCharm

#K-Means聚类算法

if __name__ == "__main__":
    # K-Means聚类算法

    import pandas as pd
    from sklearn.cluster import KMeans  # 导入K均值聚类算法

    inputfile = 'zscoreddata.xls'  # 待聚类的数据文件
    k = 5  # 需要进行的聚类类别数

    # 读取数据并进行聚类分析
    data = pd.read_excel(inputfile)  # 读取数据

    # 调用k-means算法，进行聚类分析
    kmodel = KMeans(n_clusters=k, n_jobs=4)  # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data)  # 训练模型

    print kmodel.cluster_centers_  # 查看聚类中心
    print kmodel.labels_  # 查看各样本对应的类别