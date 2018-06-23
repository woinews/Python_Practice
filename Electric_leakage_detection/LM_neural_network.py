#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r'electricity_model.xls')
data = data
#划分训练数据和测试数据，20%作为训练数据，80%作为测试数据
p = 0.8 #设置训练数据比例 
train = data.iloc[:int(len(data)*p),:]
test = data.iloc[int(len(data)*p):,:]

x = train.iloc[:,:3].as_matrix()
y = train.iloc[:,3].as_matrix()

x_test = test.iloc[:,:3].as_matrix()
y_test = test.iloc[:,3].as_matrix()

#构建LM神经网络模型
from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Activation #导入神经网络层函数、激活函数

net = Sequential() #建立神经网络
net.add(Dense(input_dim = 3, units = 10)) #添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Activation('relu')) #隐藏层使用relu激活函数
net.add(Dense(input_dim = 10, units = 1)) #添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Activation('sigmoid')) #输出层使用sigmoid激活函数
net.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) #编译模型，使用adam方法求解
#二分类问题损失函数为binary_crossentropy
#adam是梯度下降算法，一种默认的高效算法
#使用metrics=['accuracy']来输出准确率

net.fit(x, y, nb_epoch=1000, batch_size=50) #训练模型，循环1000次,batch_size每次训练和梯度更新块的大小为1（每次只训练一个样本）
net.save_weights(r'net.model') #保存模型
predictions = net.predict(x) #对数据进行预测，将训练模型运用于数据集x
train['预测值'] = [int(np.round(x)) for x in predictions]

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

predict_result = net.predict_classes(x).reshape(len(y)) #预测结果变形,作为混淆矩阵函数的输入yp
#这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n
cm_plot(y, predict_result).show() #显示混淆矩阵可视化结果
scores = net.evaluate(x, y)
print("训练样本准确率：%s: %.2f%%" % (net.metrics_names[1], scores[1]*100)) #计算模型准确率

from sklearn.metrics import roc_curve #导入ROC曲线函数

predict_result = net.predict(x_test).reshape(len(y_test))
scores_test = net.evaluate(x_test, y_test)
print("测试样本准确率：%s: %.2f%%" % (net.metrics_names[1], scores_test[1]*100)) #计算模型准确率

fpr, tpr, thresholds = roc_curve(y_test, predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of LM') #作出ROC曲线
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果

