# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:38:28 2018

@author: inews
""" 

import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = 'D:\数据分析\销售数据\销售明细'
os.chdir(path)
tips = pd.read_excel('销售明细.xlsx')

import time 
from datetime import datetime


def difftime(date_time):            
    time_delta=datetime(2017,12,31)-max(date_time)
    return time_delta

rfmdata = tips.groupby('用户ID').agg({'日期':difftime,'销售额(RMB)':['count','sum']})
rfmdata.columns = ['时间间隔','频率','销售额']
rfmdata['时间间隔'] = rfmdata['时间间隔'].dt.days

k = 3
iteration = 500
rfmdata_zs = 1.0*(rfmdata - rfmdata.mean())/rfmdata.std()
model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
model.fit(rfmdata_zs)

r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis = 1)
r.columns = list(rfmdata.columns)+[u'类别数目']
print(r)

q = pd.concat([rfmdata,pd.Series(model.labels_,index=rfmdata.index)], axis = 1)  #若数据集只有一个元素时，会报错
q.columns = list(rfmdata.columns) +[u'聚类类别']
q.to_excel('outputfile.xlsx') #保存结

def density_plot(data): #自定义作图函数
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
    [p[i].set_ylabel(u'密度') for i in range(k)]
    plt.legend()
    return plt


pic_output = 'xxx' #概率密度图文件名前缀
for i in range(k):
    density_plot(rfmdata[q[u'聚类类别']==i]).savefig(u'%s%s.png' %(pic_output, i))
