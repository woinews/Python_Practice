#-*- coding: utf-8 -*- 
#对数据进行基本的探索
#返回缺失值个数以及最大最小值

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data = pd.read_csv(r'D:\datasheet\Customer_cluster\data\air_data.csv', encoding = 'utf-8') 

explore = data.describe(percentiles = [], include = 'all').T #包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
explore['null'] = data.isnull().sum()
#explore['null'] = len(data)-explore['count'] #describe()函数自动计算非空值数，需要手动计算空值数

explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值'] 
'''这里只选取部分探索结果。
describe()函数自动计算的字段有
count（非空值数）、unique（唯一值数）、
top（频数最高者）、freq（最高频数）、
mean（平均值）、std（方差）、
min（最小值）、50%（中位数）、max（最大值）'''
print('数据总量：%s' % len(data))
print(explore)
#explore.to_excel('explore_result.xlsx') #导出结果

'''从探索结果可以看出，总数据有62988条，其中第一年总票价SUM_YR_1和第二年总票价SUM_YR_2均
存在缺失值且最小值为0，可能是客户取消航班所致，这部分数据需要筛选出来；观测窗口总飞行公里
数SEG_KM_SUM不存在缺失值且最小值不为0，说明该数据的所有客户记录均有乘机记录，符合规则；平
均折扣率avg_discount虽然无缺失值，但最小值为0，有可能是因为顾客乘坐的航班均是无折扣的，符
合业务解释
根据以上分析，现做如下筛选
SUM_YR_1或SUM_YR_2 缺失或者均为0的'''
#计算票价缺失值的数据条数
SUM_YR_missing = data.loc[(data['SUM_YR_1'].isnull()|data['SUM_YR_2'].isnull())]
SUM_YR_missing_count = len(SUM_YR_missing)

#计算SUM_YR_1和SUM_YR_2均为0的数据条数
SUM_YR_zero = data.loc[(data['SUM_YR_1'] == 0) & (data['SUM_YR_2'] == 0)]
SUM_YR_zero_count = len(SUM_YR_zero)
# percent of data that is missing/0
SUM_YR_missing_perc = (SUM_YR_missing_count + SUM_YR_zero_count)/len(data)
print('票价缺失或者均为0的数据条数占比为：%.2f%%' % SUM_YR_missing_perc)
'''可以得知，这个占比非常小，对问题影响不大，可以舍弃这部分数据'''
data_cleaning = data.loc[~(data['SUM_YR_1'].isnull()|data['SUM_YR_2'].isnull())]
data_cleaning = data_cleaning.loc[~((data['SUM_YR_1'] == 0) & (data['SUM_YR_2'] == 0))]
print('数据清洗之后剩余数据条数为 %s' % len(data_cleaning))

#数据降维，根据分析需求选取属性
data_request = data_cleaning[['FFP_DATE','LOAD_TIME','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

'''
FFP_DATE:入会时间
LOAD_TIME:观测窗口的结束时间
LAST_TO_END:最后一次乘机时间至观察窗口末端时长
FLIGHT_COUNT:飞行次数
SEG_KM_SUM:观测窗口总飞行公里数
avg_discount:平均折扣率
'''

#数据变换，根据算法需要，对数据进行属性构造
'''
本算法将客户关系长度L，最近消费时间间隔R，消费频率F，飞行里程M和折扣均值C作为客户价值指标
客户关系长度L = LOAD_TIME - FFP_DATE
最近消费时间间隔R = LAST_TO_END
消费频率F = FLIGHT_COUNT
飞行里程M = SEG_KM_SUM
折扣均值C = avg_discount
'''

#数据表中的日期均为object类型，需要转换为标准格式的日期datetime
data_request['LOAD_TIME'] = pd.to_datetime(data_request['LOAD_TIME'], format = "%Y/%m/%d")
data_request['FFP_DATE'] = pd.to_datetime(data_request['FFP_DATE'], format = "%Y/%m/%d")

data_request['L'] = data_request['LOAD_TIME'] - data_request['FFP_DATE']

data_convert = data_request[['L','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

data_convert.rename(columns={'LAST_TO_END':'R', 'FLIGHT_COUNT':'F', 'SEG_KM_SUM':'M','avg_discount':'C'}, inplace = True)

data_convert_explore = data_convert.describe(percentiles = [], include = 'all').T

print("data_convert_explore") #观察数据转换之后各属性的取值情况

#数据标准化：因为5个指标的取值范围差异较大，需要进行数据标准化，消除量级的影响

data_convert.L = data_convert.L.dt.days.astype(int)

data_zscore = (data_convert - data_convert.mean(axis = 0))/(data_convert.std(axis = 0))
data_zscore.columns = ['Z' +i for i in data_zscore.columns]

print('z标准化之后的数据：\n' , data_zscore.head())

#模型构建：使用K-means聚类算法对客户进行分群
print('-------------------进行聚类分析---------------')
from sklearn.cluster import KMeans #导入K均值聚类算法
k = 5  #需要进行的聚类类别数
#调用k-means算法，进行聚类分析
model = KMeans(n_clusters = k, init='k-means++', max_iter=300, n_jobs = 1) #n_jobs是并行数，一般等于CPU数较好
model.fit(data_zscore) #训练模型

r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis = 1)
r.columns = list(data_zscore.columns)+[u'类别数目']
print('聚类分群结果：\n', r)

cluster_result = pd.concat([data_zscore,pd.Series(model.labels_,index=data_zscore.index)], axis = 1)  #若数据集只有一个元素时，会报错
cluster_result.columns = list(data_zscore.columns) +[u'聚类类别']
cluster_result.to_excel('cluster_result.xlsx') #保存结果
#data_cleaning['聚类类别'] = cluster_result.聚类类别  #详细数据
#data_convert['聚类类别'] = cluster_result.聚类类别   #未标准化前的聚类数据
#data_convert.to_excel('result_data.xlsx')

#绘制概率密度图（绘制原始数据的概率分布图，更方便观察）
def density_plot(data): #自定义作图函数
    p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False, figsize = (8,8) )
    [p[i].set_ylabel(u'密度') for i in range(k)]
    plt.legend()
    plt.show()
    return plt

pic_output = 'cluster_result' #概率密度图文件名前缀
for i in range(k):
    density_plot(data_convert.loc[:,'L':'C'][cluster_result[u'聚类类别']==i]).savefig(u'%s%s.png' %(pic_output, i))


#客户价值分析：结合业务对各聚类群体进行定义

#模型评价，使用TSNE进行数据降维，查看聚类结果分布情况

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, learning_rate=100, init='pca', random_state=0)
tsne_data = tsne.fit_transform(data_zscore) #进行数据降维
tsne = pd.DataFrame(tsne_data, index = data_zscore.index) #转换数据格式
plt.figure()
for i in range(k):
    d = tsne[cluster_result[u'聚类类别'] == i]
    plt.plot(d[0], d[1], '.')

#使用PCA进行数据降维并可视化
from sklearn.decomposition import PCA

pca = PCA()
data = pca.fit_transform(data_zscore)
data = pd.DataFrame(data,index=data_zscore.index)
for i in range(k):
    d = data[cluster_result[u'聚类类别'] == i]
    plt.plot(d[0], d[1], '.')


