#-*- coding: utf-8 -*-
#Python数据分析与挖掘实战 第五章 使用随机逻辑回归筛选特征值


import pandas as pd

filename = r'D:\DataAnalysis\Python_practice\chapter5\demo\data\bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8]#.as_matrix() #选取自变量，在书中将df转化为矩阵（.as_matrix）进行运算。而本程序使用的参数可以为dataframe，故可以不转化
y = data.iloc[:,8]#.as_matrix() #loc是根据条件选取，iloc是根据索引进行选取切片

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR 

rlr = RLR() 
# 建立随机逻辑回归模型，筛选变量 
# 可以使用参数设置阈值： selection_threshold = 0.5 ，默认0.25(即得分<0.25的特征值会被剔除)

rlr.fit(x, y) #训练模型

rlr.get_support() # 获取特征筛选结果，也可以通过 .scores_方法获取各个特征的分数 

filter_columns = data.columns[0:8][rlr.get_support()] #选取特征字段数据
print(u'---------start-----------')
print(u'有效特征为: %s' % ','.join(filter_columns))
x = data[filter_columns]#.as_matrix(columns=None)
lr = LR()  # 建立逻辑回归模型 

lr.fit(x, y)  # 用筛选后的特征数据来训练模型

print(u'---------end-----------')
print(u'模型的平均正确率为%s' % lr.score(x, y)) 



