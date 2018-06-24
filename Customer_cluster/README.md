
# 客户价值分析   
## 一、建模目标  
  * 借助航空公司客户数据，对客户进行分群   
  * 对不同的客户群体进行特征分析，比较不同客户群体的客户价值  
  * 对不同价值的客户群体提供个性化服务，实现精准营销  
 
## 二、分析方法与过程  
  * 收集原始数据，取样（历史数据、增量数据）  
  * 数据探索分析（数据清洗、缺失值处理、数据降维、数据变换） 
  * 建立模型，模型分析，客户价值分类

## 三、聚类结果解读 
  * 聚类结果：  
  ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result.png)  
    *  第0类客户群体，其分布概率图如下：  
        ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result_0.png)  
    *  第1类客户群体，其分布概率图如下：  
        ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result_1.png)
    *  第2类客户群体，其分布概率图如下：  
        ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result_2.png)
    *  第3类客户群体，其分布概率图如下：  
        ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result_3.png)
    *  第4类客户群体，其分布概率图如下：  
        ![image](https://github.com/woinews/Python_Practice/blob/master/Customer_cluster/cluster_result/cluster_result_4.png)
    *  第0类客户群体最近一次消费R较低，其消费频率F和里程M对比第。第1类客户群体的R值很高，较长时间没有乘坐航班，其余的数值都很低，可认定为一般价值客户。
