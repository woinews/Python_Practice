
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
    *  第0类客户群体在入会时间L、评价折扣率C属性上是最小值，其乘坐次数F和里程M都较低，代表这是一般价值客户；  
    *  第1类客户群体在F，M上最大，且R很低，说明这是一类重要客户，且创造的价值较高，可进行一对一营销，提高这部分客户的忠诚度和满意度，尽可能保持其较高的消费水平，是重要保持客户；  
    *  第2类客户在R上达到最大，且F，M属性上最小，代表这是一类低价值的用户，可能在打折促销时才会产生消费行为。  
    *  第3类客户在L上达到最大，F和M较高，代表这类客户其入会时间长，但是消费能力尚可，总结就是该客户群体的不确定性很高但是有价值，值得维护与这类客户的关系，需要推测用户消费的异常状况，主动联系，延长客户的生命周期。
    *  第4类客户的C属性最大，但F，M较低，入会时间L较短，可看出这类客户群体的潜在价值较高，应该努力发展这类客户的消费能力，提升其价值，属于重要发展客户。
