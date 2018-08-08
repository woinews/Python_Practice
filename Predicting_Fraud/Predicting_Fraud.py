# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 10:48:04 2018
@author: woinews
金融交易服务欺诈行为预测，数据量为636万，而欺诈行为占比很低，正负样本极度不平衡，故采取了xgboost进行分类预测，观察AUPRC曲线覆盖面积对模型进行评价。
数据集链接：https://www.kaggle.com/ntnu-testimon/paysim1/downloads/PS_20174392719_1491204439457_log.csv/2
代码参考自:https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services
 """
import pandas as pd
import numpy as np
df = pd.read_csv(r'D:\datasheet\PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig','oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
pd.set_option('display.max_columns',20)  #避免输出时省略了中间的字段，可以将展示的列数设置在0-20
print(df.head())  
#对数据进行探索性分析
explore = df.describe(percentiles = [], include = 'all').T
explore['null'] = df.isnull().sum()
explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值']
print('数据总量：%s' % len(df))
print(explore)   #可以看出数据集中没有空值
#查看有诈骗行为的都是以什么方式进行交易的,可以发现只有其中的TRANSFER和CASH_OUT会出现诈骗行为
print(df.loc[df.isFraud == 1].type.drop_duplicates())
#查看TRANSFER和CASH_OUT各发生了多少次诈骗行为
print(df.loc[df.isFraud == 1].groupby('type')['type'].count())
'''
对所有特征进行筛选
step:在现实世界中映射的时间单位，保留
type:只有其中的TRANSFER和CASH_OUT会出现诈骗行为,故只选择这两种type的数据进行建模
amount：本币交易金额，保留
nameOrig：开始交易的客户，名字应该没有特殊含义，舍去
oldBalanceOrig：客户交易前的初始余额，保留
newBalanceOrig：客户交易后的账户余额，保留
nameDest：接收方ID，ID应该无特殊含义，舍去
oldBalanceDest：接收方在交易前的初始余额
newBalanceDest：接收方在交易后的余额
isFraud：是否诈骗，标签
isFlaggedFraud：被标记的人是否试图转移超过20W美金的诈骗行为，这个含义暂时未知，可以考察该字段与isFraud的关系
'''
print('isFlaggedFraud为1时的交易数：%s' %len(df.loc[df.isFlaggedFraud == 1]))
print('isFlaggedFraud为1时isFraud也为1的交易数：%s' %len(df.loc[(df.isFlaggedFraud == 1)&(df.isFraud == 1)]))
#说明所有的isFlaggedFraud标记为1时，isFraud也是1
print('isFraud为1时的交易数：%s' %len(df.loc[df.isFraud == 1]))
#16/8223 占比太小，不能看出关系
#找出isFlaggedFraud为1时的type类型有哪些，可以发现只有TRANSFER
print("isFlaggedFraud为1时的type类型有：%s" %list(df.loc[df.isFlaggedFraud == 1].type.drop_duplicates()))
#isFlaggedFraud为1时最小的交易金额
print("isFlaggedFraud为1时最低交易金额：$%s" %(df.loc[df.isFlaggedFraud == 1].amount.min()))
#isFlaggedFraud为0时最大的交易金额
print("isFlaggedFraud为0时最高交易金额：$%s" %(df.loc[df.isFlaggedFraud == 0].amount.max()))
#可以看出isFlaggedFraud似乎并不是根据金额大小来进行标记的
#总结 ：舍弃isFlaggedFraud
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
randomState = 5
np.random.seed(randomState)
Y = X['isFraud']
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
#TRANSFER和CASH_OUT是分类变量，刚好用0和1进行替换
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)
#考察收款方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据与isFraud的关系
Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]
Xfraud_BalanceDest_iszero = Xfraud.loc[(Xfraud.oldBalanceDest == 0) &
                                       (Xfraud.newBalanceDest == 0) & (Xfraud.amount != 0)]
print('发生欺诈事件的接收方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：%s' %((len(Xfraud_BalanceDest_iszero)/len(Xfraud))))
XnonFraud_BalanceDest_iszero = XnonFraud.loc[(XnonFraud.oldBalanceDest == 0) & 
                                             (XnonFraud.newBalanceDest == 0) & (XnonFraud.amount != 0)]
print('未发生欺诈事件的接收方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：%s' %((len(XnonFraud_BalanceDest_iszero)/len(XnonFraud))))
'''
发生欺诈事件的接收方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：0.4955
未发生欺诈事件的接收方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：0.00062
可以看出这两个事件有很强的关系，接收方账户为0时有49.55%会导致最后产生欺诈行为
将接收方目标账户为0的值替换为-1，对机器学习算法检测欺诈行为可能会更准确
'''
X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0),
      ['oldBalanceDest', 'newBalanceDest']] = - 1
#考察交易方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据与isFraud的关系
Xfraud_BalanceOrig_iszero = Xfraud.loc[(Xfraud.oldBalanceOrig == 0) & 
                                   (Xfraud.newBalanceOrig == 0) & (Xfraud.amount != 0)]
print('发生欺诈事件中发起方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：%s' %((len(Xfraud_BalanceOrig_iszero)/len(Xfraud))))
XnonFraud_BalanceOrig_iszero = XnonFraud.loc[(XnonFraud.oldBalanceOrig == 0) & 
                                         (XnonFraud.newBalanceOrig == 0) & (XnonFraud.amount != 0)]
print('未发生欺诈事件中发起方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：%s' %((len(XnonFraud_BalanceOrig_iszero)/len(XnonFraud))))
'''
发生欺诈事件中发起方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：0.00304
未发生欺诈事件中发起方交易前初始余额为0，交易后余额也为0，但交易金额不为0的数据占比为：0.4737
与正常的交易（47.37%）相比，发生欺诈事件（0.003%）的比例要小得多
将发起方账户余额为0的值全部替换为空值（便于机器学习算法进行区分）
'''
X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0),
      ['oldBalanceOrig', 'newBalanceOrig']] = np.nan
      
#设置两个新特征值，用于区分问题数据
      
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest    
 #可视化
import matplotlib.pyplot as plt
import seaborn as sns
limit = len(X)
#制作分布散点图
#stripplot的作图原理就是按照x属性所对应的类别分别展示y属性的值，适用于分类数据
def plotStrip(x, y, hue, figsize = (14, 9)):    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, \
             hue = hue, jitter = 0.4, marker = '.', \
             size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), \
               loc=2, borderaxespad=0, fontsize = 16);
    return ax
ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 16)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent \
transactions over time', size = 20)
#或者使用分布函数方法进行绘图
#ax = sns.stripplot(x='isFraud', y='step', hue='type', data=X, order=None, 
#                  hue_order=None, jitter=True, 
#                  split=None, orient=None, 
#                  color=None, palette=None, size=5, 
#                  edgecolor="gray", linewidth=2, 
#                  ax=None)
#'''
#x: X轴数据
#y: Y轴数据
#hue: 区分不同种类数据的column name
#data: DataFrame类型数据
#jitter: 将数据分开点，防止重叠
#'''
#可以看出随时间分布，欺诈事件的分布更为均匀，而且在正常事件中CASH-OUT的数量明显比TRANSFER多
#绘制出不同事件随金额amount的分布散点图
limit = len(X)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine \
and fraudulent transactions over amount', size = 18)
#绘制出不同事件随errorBalanceDest的分布散点图
limit = len(X)
ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('- errorBalanceDest', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in \
destination account balances', size = 18)
#对比以上两个图可以发现，errorBalanceDest比单纯使用amount更能用于区分欺诈和真实事件
#通过绘制3D散点分布图
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D  #用于绘制3D图形
x = 'errorBalanceDest'
y = 'step'
z = 'errorBalanceOrig'
zOffset = 0.02
limit = len(X)
sns.reset_orig() # prevent seaborn from over-riding mplot3d defaults
fig = plt.figure(figsize = (10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], -np.log10(X.loc[Y == 0, z][:limit] + zOffset), c = 'g', marker = '.', \
  s = 1, label = 'genuine')    
ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \-np.log10(X.loc[Y == 1, z][:limit] + zOffset), c = 'r', marker = '.', \
  s = 1, label = 'fraudulent')
ax.set_xlabel(x, size = 16); 
ax.set_ylabel(y + ' [hour]', size = 16); 
ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)
ax.set_title('Error-based features separate out genuine and fraudulent transactions', size = 20)
plt.axis('tight')
ax.grid(1)
noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.',
                          markersize = 10, label='genuine')
fraudMarker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.',
                          markersize = 10, label='fraudulent')
plt.legend(handles = [noFraudMarker, fraudMarker], \
           bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16})
#从3D分布可以看出step对isFraud的区分是没有关系的，可以将该特征删除
#使用热力图对每个特征之间的相关关系进行考察
Xfraud = X.loc[Y == 1] 
XnonFraud = X.loc[Y == 0]
                  
correlationNonFraud = XnonFraud.loc[:, X.columns != 'step'].corr() 
#去除step后对各特征值的相关性进行考察,返回一个9X9的矩阵
mask = np.zeros_like(correlationNonFraud)   #返回一个9X9的全0矩阵
indices = np.triu_indices_from(correlationNonFraud)  #返回一个9X9的上三角矩阵（主对角线以上）
mask[indices] = True  #将9X9的上三角矩阵的值都设置为1
grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))
cmap = sns.diverging_palette(220, 8, as_cmap=True)
ax1 =sns.heatmap(correlationNonFraud, ax = ax1, vmin = -1, vmax = 1, \
    cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)
ax1.set_xticklabels(ax1.get_xticklabels(), size = 16)
ax1.set_yticklabels(ax1.get_yticklabels(), size = 16)
ax1.set_title('Genuine \n transactions', size = 20)
correlationFraud = Xfraud.loc[:, X.columns != 'step'].corr()
ax2 = sns.heatmap(correlationFraud, vmin = -1, vmax = 1, cmap = cmap, \
ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False, \
    cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', \
                                 'ticks': [-1, -0.5, 0, 0.5, 1]})
ax2.set_xticklabels(ax2.get_xticklabels(), size = 16)
ax2.set_title('Fraudulent \n transactions', size = 20)
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14)
'''
正常的交易行为，amount与errorBalanceOrig有很强的相关性，而欺诈行为中，amount与errorBalanceOrig的相关性
很弱，而与oldBalanceOrig的相关性很强，因此errorBalanceOrig特征的设定可以很好地用于区分正常交易与欺诈行为
'''
print('skew = {}'.format( len(Xfraud) / float(len(X)) ))

'''可以看出欺诈事件与正常事件的数量差别很大，
采用precision-recall curve (AUPRC) 【即查准率-查全率曲线】
PR曲线在正负样本比例悬殊较大时更能反映分类的性能
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
#区分测试集和训练集，使用XGBClassifier对数据进行分类并计算AUPRC
#对于XGBoost的使用查看连接 https://blog.csdn.net/qunnie_yi/article/details/80129857
X = X.drop(['isFraud'], axis = 1)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, 
                                                random_state = randomState)
# Long computation in this cell (~1.8 minutes)
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())  #计算权重
clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
probabilities = clf.fit(trainX, trainY).predict_proba(testX)
print('AUPRC = {}'.format(average_precision_score(testY, probabilities[:, 1])))
#输出该训练模型的重要特征值（从高到低排序）
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)
colours = plt.cm.Set1(np.linspace(0, 1, 9))
#plot_importance(clf)
ax = plot_importance(clf, height = 1, color = colours, grid = False, 
                     show_values = False, importance_type = 'cover', ax = ax)
plt.show()
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
 
cm_plot(testY, clf.predict(testX)).show()
from sklearn.metrics import accuracy_score #用于输出混淆矩阵中分类正确的事件比例
score = accuracy_score(testY,clf.predict(testX))
print("模型准确率: %.2f%%" % (score*100))
