#-*- coding: utf-8 -*-
#拉格朗日插值代码
import pandas as pd 
from scipy.interpolate import lagrange #导入拉格朗日插值函数

#读取数据,该数据没有列名，从第一行开始，需设置参数header=None
data = pd.read_excel('../data/missing_data.xls', header=None) 

print(data)  #先观察3个用户一个月内的用电量数据，发现有缺失值，需要进行填补

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  y = s.reindex(list(range(n-k, n)) + list(range(n+1, n+1+k))) 
  #取数，有可能会取到空值。另外这里用series[list(range(x))]时会出现警告，修改为series.reindex(list(range(x)))
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果n

#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i].isnull())[j]: #如果为空即插值
      data[i][j] = ployinterp_column(data[i], j)

print(data)  #观察插值之后的用电量数据







