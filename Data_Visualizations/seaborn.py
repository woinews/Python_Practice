
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:21:02 2018

@author: chenzx
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

sns.set(style='white', color_codes=True)

iris = pd.read_csv(r'D:\datasheet\Iris.csv')
pd.set_option('display.max_columns',20) #避免输出时省略了中间的字段，可以将展示的列数设置在0-20
print(iris.head())

print("\n查看每个种类各有多少样本:\n", iris["Species"].value_counts())
#plot默认生成曲线图，kind参数可选的值为：line, bar(柱状图), barh, kde, density, scatter（散点图）

from pylab import mpl       #plot输出图形中文设置问题
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt1 = iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt1.set_title('使用plot绘制花萼长度与花萼宽度的散点图（cm）')

# seaborn 的 jointplot 函数可以在同一个图中画出二变量的散点图和单变量的柱状图
ax1 = sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)

# 上面的两个散点图并不能显示每一个点所属的类别
# 可以使用 seaborn 的 FacetGrid 函数按照Species花的种类来在散点图上标上不同的颜色
ax2 = sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
#通过箱线图来查看单个特征的分布，查看不同花种中花瓣长度的分布

ax3 = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

#通过stripplot可以使点落在箱线图上，注意此处要将坐标图用ax4先保存起来，这样第二次才会在原来的基础上加上散点图
ax4 = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax4 = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

# violinplot 小提琴图，查看密度分布，结合了前面的两个图，并且进行了简化
# 数据越稠密越宽，越稀疏越窄

ax5 = sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)

#核密度图对于观察单变量的关系很有作用，下图为不同种类花瓣长度的关系
ax6 = sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()

#通过交叉图可以查看两个变量间的关系
ax7 = sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)

#交叉图默认展示的是柱状图，可以通过修改diag_l=kind参数更改图形类别
ax8 = sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")

#通过pandas内置的plot可以快速绘制散点图
plot1 = iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

#pandas另一个强大的绘图功能是安德鲁斯曲线
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")







