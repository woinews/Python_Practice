# python数据挖掘
## 1. k-means.py  
    用k-means对用户消费进行分群，以RFM为依据  

## 2. logistic_regression.py  
    使用logistic回归分析银行贷款违约情况与客户特征的关系，使用随机逻辑回归筛选特征值  
 
## 3. decision_tree.py & tree.dot   
    使用ID3算法的决策树模型，基于信息熵的算法，只能处理离散属性的数据  

## 4. keras_binary_network.py  
    对二分类输入输出变量，使用人工神经网络预测销量高低
    
## 5. apriori_for_python3.py  
    书本中的代码对结果输出的解释不够完整，我参考了一些网站对apriori的代码实现，最后在http://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/ 中找到了能在Python3上运行的且结果较为完整的代码。这里保存以作后续有需求时使用。  
    同时有一篇很好的中文博客说明：https://blog.csdn.net/eastmount/article/details/53368440
    关联规则的步骤：  
        a.找出所有频繁项集（>=最小支持度的项集）  
        b.由频繁项集产生强关联规则，这些规则必须大于或者等于最小支持度和最小置信度。  
    Apriori算法的核心：频繁项集的子集必为频繁项集，非频繁项集的超集一定是非频繁项集
  
## 6.arima_model.py
    ARIMA模型建模操作：
    平稳性检验---白噪声检验---是否差分（几阶差分趋近平稳）---AIC和BIC指标值（BIC信息量达到最小时对应的p和q）---模型定阶---预测
















