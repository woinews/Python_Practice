#-*- coding: utf-8 -*-
#使用ID3决策树算法预测销量高低
import pandas as pd

inputfile = r'D:\DataAnalysis\Python_practice\chapter5\demo\data\sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'序号') 

#数据是类别标签，要将它转化为数值
#用1来表示“好、是、高”，用-1来表示“坏、否、低”
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
data = data.astype(int)   #将数据类型转换为整型
x = data.iloc[:,:3]  #选取自变量
y = data.iloc[:,3]  #选取因变量

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x, y) #训练模型

#可视化决策树
#将生成的决策树保存在.dot文件中
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
x = pd.DataFrame(x)
with open("tree.dot", 'w') as f:
  f = export_graphviz(dtc, feature_names = x.columns, out_file = f)

#随后将代码放在Graphviz中运行即可看到决策树
'''
为了正常显示中文，需要在代码中增加以下字段
edge [fontname = "SimHei"];
node [fontname = "SimHei"];

随后将其保存为utf-8格式
'''
#Graphviz下载地址 https://graphviz.gitlab.io/_pages/Download/Download_windows.html
#Graphviz 使用说明 https://blog.csdn.net/frankax/article/details/77035397








