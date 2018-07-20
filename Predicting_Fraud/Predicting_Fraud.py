# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:36:23 2018

@author: chenzx
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
step:在现实世界中映射的时间单位，无特殊意义舍去
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
      
      
      
      
      
      
      
      
      
      
      
      





