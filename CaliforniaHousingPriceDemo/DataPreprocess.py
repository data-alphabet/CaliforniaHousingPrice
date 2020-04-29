# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:32:06 2020

@author: Administrator
"""


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from FeatureEngineering import CombineAttributesAdder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# 拆分数据集为数据部分和标签部分
def get_data_labels(train_set, test_set, target_column='median_house_value'):
    # 训练集数据
    train_data = train_set.drop(target_column, axis=1)
    # 训练集标签
    train_labels = train_set[target_column].copy()
    # 测试集数据
    test_data = test_set.drop(target_column, axis=1)
    # 测试集标签
    test_labels = test_set[target_column].copy()
    return train_data, train_labels, test_data, test_labels

# 1.缺失值处理
# # total_bedrooms
# # 计算中位数
# median_total_bedrooms = data['total_bedrooms'].median()
# # 用中位数填充缺失值
# # pandas.DataFrame.fillna() Fill NA/NaN values using the specified method.
# data['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
    
# # 删除文字属性 ocean_proximity
# data_num = data.drop('ocean_proximity', axis=1)

# # 创建 imputer 对象 缺失值处理器
# # sklearn.impute.SimpleImputer() Imputation on transformer for completing missing values.
# imputer = SimpleImputer(strategy='median')

# # 为所有属性生成填充策略
# imputer.fit(data_num)

# # 查看每个属性要替换的值
# imputer.statistics_

# # 完成填充，结果是Numpy数组
# X = imputer.transform(data_num)

# # 将Numpy数组转换回DataFrame格式
# data_num_tr = pd.DataFrame(X, columns=data_num.columns)
# data_num_tr.info()

# # 2.处理文本和类别属性
# encoder = LabelEncoder()
# data_category = data['ocean_proximity']
# data_category_encoded = encoder.fit_transform(data_category)

# # 查看转换器学习的映射关系
# encoder.classes_
# # 查看替换结果
# data_category_encoded.reshape(-1, 1)
    
# # 3.one-hot 编码
# # OneHotEncoder 转换器
# encoder = OneHotEncoder(category='auto')
# data_category_1hot = encoder.fit_transform(data_category_encoded.reshape(-1, 1))
# # 返回的结果是一个稀疏矩阵，为节省内存空间，可以调用 .toarray() 方法，将其转换为Numpy数组
# data_category_1hot.toarray()

# # LabelBinarizer 转换器
# # 一次性将文本转换为独热编码
# encoder = LabelBinarizer()
# data_category_1hot = encoder.fit_transform(data_category)
# data_category_1hot

# # 4.特征缩放
# # 归一化与标准化
# num_pipline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('attribs_addr', CombineAttributesAdder()),
#     ('std_scaler', StandardScaler())
#     ])

# data_num_tr = num_pipline.fit_transform(data_num)
# # 转化为DataFrame
# columns = list(data_num.columns) + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
# data_num_tr = pd.DataFrame(data_num_tr, columns=columns)
# data_num_tr.head()

# 5.使用流水线处理
# 数据集选择器
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_name].values

# 文本独热编码转换器
class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder
    
    def fit(self, X, Y=None):
        self.encoder.fit(X)
        return self
    
    def transform(self, X, Y=None):
        return self.encoder.transform(X)

# 数据预处理流水线
#
def SuperPipeline(data):
    data_num = data.drop(['ocean_proximity'], axis=1)
    data_cat = data['ocean_proximity'].copy()
    
    # FeatureUnion 类，可以将多个流水线合并在一起
    num_attribs = list(data_num.columns)
    # cat_attribs = list(data_cat.columns)
    cat_attribs = 'ocean_proximity'
    label_encoder = LabelBinarizer()
    
    # 数值处理
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_addr', CombineAttributesAdder()),
        ('std_scaler', StandardScaler())
        ])
    
    # 文本处理
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', MyLabelBinarizer(label_encoder))
        ])
    
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
        ])
    
    # 运行整条流水线
    data_prepared = full_pipeline.fit_transform(data)
    return data_prepared
