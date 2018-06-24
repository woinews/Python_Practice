
# 电力窃漏电用户自动识别   
## 一、建模目标  
  * 归纳出窃漏电用户的关键特征，构建窃漏电用户的识别模型  
  * 利用实时监测数据，调用模型实现实时诊断  
  
## 二、分析方法与过程  
  * 收集原始数据，取样（历史数据、实时数据）  
  * 数据探索分析（数据清洗、缺失值处理、数据变换） 
  * 建模&诊断（训练模型，模型评分，自动诊断） 

## 三、结果诊断  
  * LM神经网络模型预测结果的混淆矩阵图如下：  
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/LM_net_confusion_matrix.png)  
    LM神经网络模型准确率为 92.67%  
    用LM神经网络对测试数据进行预测，可得如下矩阵图：  
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/LM_net_test_confusion_matrix.png)  
    可得诊断准确率为 94.92%  
    绘制其ROC曲线进一步判断模型的分类效果，可以看出效果较好:  
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/LM_net_ROC.png)
    
  * CART决策树模型预测结果的混淆矩阵图如下：    
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/CART_tree_confusion_matrix.png)  
    CART决策树模型准确率为 94.40%  
    用CART决策树模型对测试数据进行预测，可得如下矩阵图：  
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/CART_tree_confusion_matrix_test.png)  
    可得诊断准确率为 94.92%
    绘制其ROC曲线进一步判断模型的分类效果，可以看出效果较好:  
    ![image](https://github.com/woinews/Python_Practice/blob/master/Electric_leakage_detection/CART_tree_ROC.png)  
    
  * LM神经网络模型和CART决策树模型的分类效果都很好，可能是因为样本数据较少，两者的预测结果一样，但从处理速度上看，CART决策树更快，所以选取CART决策树模型对用电用户进行实时诊断会更好。
    
## 四、使用到的方法说明
  * 拉格朗日插值法：利用已知的点建立拉格朗日插值函数，由对应点x求出函数值y进行近似代替
  【 如何直观地理解拉格朗日插值法？ - 马同学的回答 - 知乎https://www.zhihu.com/question/58333118/answer/262507694 】 
  * CART决策树解释：https://blog.csdn.net/suipingsp/article/details/42264413
  
