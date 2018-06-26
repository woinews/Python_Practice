# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:43:34 2018

@author: chenzx
"""
import numpy as np
import pandas as pd

df = pd.read_csv(r'D:\datasheet\customer_churn\customer_churn.csv')

#数据探索
df.head()
explore = df.describe(percentiles = [], include = 'all').T 
explore['null'] = df.isnull().sum()
explore = explore[['count','null','unique', 'max', 'min']]
print(explore)
#可以看出，该数据集中没有缺失值
'''
特征筛选
    RowNumber：行号，这个肯定没用，删除
    CustomerID：用户编号，这个是顺序发放的，删除
    Surname：用户姓名，对流失没有影响，删除
    CreditScore：信用分数，这个很重要，保留
    Geography：用户所在国家/地区，这个有影响，保留
    Gender：用户性别，可能有影响，保留
    Age：年龄，影响很大，年轻人更容易切换银行，保留
    Tenure：当了本银行多少年用户，很重要，保留
    Balance：存贷款情况，很重要，保留
    NumOfProducts：使用产品数量，很重要，保留
    HasCrCard：是否有本行信用卡，很重要，保留
    IsActiveMember：是否活跃用户，很重要，保留
    EstimatedSalary：估计收入，很重要，保留
    Exited：是否已流失，这将作为我们的标签数据
'''
#保留重要特征
X = df.loc[:,['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]

y = df.Exited

#机器学习接收数字信息，需要将类别信息变成数值（LabelEncoder）
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
X.Geography= labelencoder1.fit_transform(X.Geography)
labelencoder2 = LabelEncoder()
X.Gender = labelencoder2.fit_transform(X.Gender)

'''
在数据探索阶段可知，Geography有3个国家，导致取值有0,1之外的数值，机器会误以为是大小关系，但实
际上它们只是类别关系，需要将数值大小转换为只有0,1表示的类别关系，OneHotEncoder可以把类别的取
值转变为多个变量组合表示

'''
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#查看转换后的第一行
X[0]

'''
所有被OneHotEncoder转换的列会排在最前面，然后才是那些保持原样的数据列（Geography排前）
第一行法国从原先的0变为(1,0,0)，根据转换规则，只需给定其中两列就可得到第三列的数值，即只需引入
其中两列的虚拟变量即可决定原变量的类别。在这里删除第0列，避免产生“虚拟变量陷阱”

“虚拟变量陷阱”：一般在引入虚拟变量时要求每一定性变量所需虚拟变量个数比该定性变量的类别数少1，即若有m个定性变量，
则只在模型中引入m-1个虚拟变量。如果引入m个虚拟变量，就会导致模型解释变量间出现完全共线性的情况。
一般将由于引入虚拟变量个数与定性因素个数相同出现的模型无法估计的问题为“虚拟变量陷阱”。
'''
#删除第0列，需要指定 axis = 1
X = np.delete(X, [0], axis = 1)

#查看转换后的第一行
X[0]

#特征矩阵处理基本完成,接下来要将对应的标签y进行转换，使其成为列向量
y = y[:, np.newaxis]

#对标签也用OneHotEncoder转换（多元决策）
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()

#将数据集的20%作为测试数据，80%作为训练数据。使用train_test_split随机划分训练集和测试集，random_state是随机数的种子
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#数据标准化处理，StandardScaler可进行均值方差归一化，即z标准化处理
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
只对特征矩阵做标准化，标签是不能动的。
训练集和测试集需要按照统一的标准变化。训练集上调用fit_transform()，其实找到了均值μ和方差σ^2，
即我们已经找到了转换规则，我们把这个规则利用在训练集上，同样，我们可以直接将其运用到测试集上
（甚至交叉验证集），所以在测试集上的处理，我们只需要标准化数据而不需要再次拟合数据
'''

#引入决策树模型
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


from sklearn.metrics import classification_report #调用classification_report模块，用以生成分析报告
print(classification_report(y_test, y_pred))

'''
precision:准确率（所有"正确被检索的item"占所有"实际被检索到的item"的比例）
recall:召回率（"正确被检索的item"占所有"应该检索到的item"的比例）
f1-score:衡量二分类模型精确度的一种指标，可以看作是模型准确率和召回率的一种加权平均
'''

#深度学习-神经网络算法
#许多时候模型过于简单带来的问题，可以通过加深隐藏层次、增加神经元的方法提升模型复杂度，加以改进
import tflearn

net = tflearn.input_data(shape=[None, 11]) 
#11是特征矩阵的列数，原本有10列，但Geography分为了两列
#shape的第一项，指的是我们要输入的特征矩阵行数，None可以让机器自动处理

#搭建3层隐藏层，每一层设置了6个神经元，激活函数为relu
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.fully_connected(net, 6, activation='relu')

#搭建输出层，用两个神经元做输出，并且使用回归方法，输出层选用的激活函数为softmax。处理分类任务的时候，softmax比较合适
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

#生成模型
model = tflearn.DNN(net)

#tflearn可视化结果
'''
打开终端，输入
tensorboard --logdir=/tmp/tflearn_logs/
然后在浏览器里输入http://localhost:6006/
'''

#训练模型
model.fit(X_train, y_train, n_epoch=30, batch_size=32, show_metric=True)
'''
    n_epoch：数据训练几个轮次。
    batch_size：每一次输入给模型的数据行数。
    show_metric：训练过程中要不要打印结果。
'''

#预测测试集的流失情况
y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test)

print('测试集准确率: %0.4f%%' % (score[0] * 100))
