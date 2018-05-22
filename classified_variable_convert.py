# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:28:02 2018

@author: woinews
"""

import pandas as pd  


# 生成数据
user_data1 = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                   'sex': ['male', 'Female', 'Female'],
                   'level': ['high', 'low', 'middle']})

# 自定义转换主过程

def convert_data(df):
    df_new = df.copy()  # 复制一份新的数据框用来存储转换结果
    for col_num, col_name in enumerate(df):  # for循环中使用enumerate方法可返回每个列的索引值和列名
        col_data = df[col_name]  # 获得每列数据
        col_dtype = col_data.dtype  # 获得每列dtype类型
        if col_dtype == 'object':  # 如果dtype类型是object（非数值型），执行条件
            df_new = df_new.drop(col_name, 1)  # 删除df数据框中要进行标志转换的列
            value_sets = col_data.unique()  # 获取分类和顺序变量的唯一值域
            for value_unique in value_sets:  # 读取分类和顺序变量中的每个值
                col_name_new = col_name + '_' + value_unique  # 创建新的列名，使用原标题+值的方式命名
                col_tmp = df.iloc[:, col_num]  # 获取原始数据列
                new_col = (col_tmp == value_unique)  # 将原始数据列与每个值进行比较，相同为True，否则为False
                df_new[col_name_new] = new_col  # 为最终结果集增加新列值
    print (df_new)  # 打印输出转换后的数据框

convert_data(user_data1)

# 对于将原始数据转换为数字进行存储的情况，可以将其替换为原来定义的字符串，再执行上面的过程
user_data2 = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                    'sex': [1, 2, 2],
                    'level': [3, 1, 2]})
    
user_data2['sex'] = user_data2['sex'].replace({1:"male",2:'Female'})   
user_data2['level'] = user_data2['level'].replace({3:"high",2:'middle',1:"low"})

convert_data(user_data2)



