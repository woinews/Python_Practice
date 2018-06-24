#-*- coding: utf-8 -*-
#构建并测试CART决策树模型

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = 'electricity_model.xls' 
data = pd.read_excel(datafile) #读取数据，数据的前三列是特征，第四列是标签

#划分训练数据和测试数据，20%作为训练数据，80%作为测试数据
p = 0.8 #设置训练数据比例

train = data.iloc[:int(len(data)*p),:]
test = data.iloc[int(len(data)*p):,:]

x = train.iloc[:,:3].as_matrix()
y = train.iloc[:,3].as_matrix()

x_test = test.iloc[:,:3].as_matrix()
y_test = test.iloc[:,3].as_matrix()

#构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型
from sklearn.metrics import accuracy_score  #用于输出模型准确率

treefile = 'tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型,模型参数https://www.cnblogs.com/pinard/p/6056319.html

tree.fit(x, y) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile) #保存模型

#定义一个用于绘制混淆矩阵图的函数
def cm_plot(y, yp):
  from sklearn.metrics import confusion_matrix #导入混淆矩阵函数  
  cm = confusion_matrix(y, yp) #混淆矩阵 
  plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens
  plt.colorbar() #颜色标签  
  
  for x in range(len(cm)): #数据标签  
    for y in range(len(cm)):
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
  
  plt.ylabel('True label') #坐标轴标签 
  plt.xlabel('Predicted label') #坐标轴标签
  return plt

predictions = tree.predict(x) #对数据进行预测，将训练模型运用于数据集x
train['预测值'] = [int(np.round(x)) for x in predictions]

cm_plot(y, tree.predict(x)).show() #cm_plot(y,y_predict)显示混淆矩阵可视化结果
#注意到Scikit-Learn使用predict方法直接给出预测结果
#模型的准确率
score = accuracy_score(y,predictions)
print("决策树模型准确率: %.2f%%" % (score*100))


#用模型预测测试样本的结果
predictions_test = tree.predict(x_test)
test['预测值'] = [int(np.round(x)) for x in predictions_test]

cm_plot(y_test, tree.predict(x_test)).show() #cm_plot(y,y_predict)显示混淆矩阵可视化结果
#模型预测测试样本的准确率
score_test = accuracy_score(y_test,predictions_test)
print("决策树模型预测测试样本的准确率: %.2f%%" % (score_test*100))

from sklearn.metrics import roc_curve #导入ROC曲线函数
#ROC详解https://blog.csdn.net/ice110956/article/details/20288239
fpr, tpr, thresholds = roc_curve(y_test, tree.predict_proba(x_test)[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') #作出ROC曲线
plt.xlabel('False Positive Rate') #误检率是相对于虚假目标的总量里有多少被误识为真实目标
plt.ylabel('True Positive Rate') #查准率是指检测到的目标里，真实目标所占的比例
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果



