# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:26:15 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from EDA import exploratory_data_analysis
from SplitDataset import randomly_split, hash_id_split, stratified_split
from FeatureEngineering import geo_data_vis, corr_matrix_vis
from DataPreprocess import get_data_labels, SuperPipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# 载入数据
data = pd.read_csv('./datasets/housing/housing.csv')
exploratory_data_analysis(data)

# 创建测试集与训练集
train_set, test_set = randomly_split(data=data, 
                            test_size=0.2, 
                            random_state=42,
                            local=False)
train_set, test_set = hash_id_split(data=data,
                                    test_size=0.2,
                                    id_column=None)
train_set, test_set = stratified_split(data=data, 
                                        test_size=0.2, 
                                        random_state=42)

# 特征工程分析
# 地理数据可视化
geo_data_vis(train_set)
corr_matrix = corr_matrix_vis(train_set)

# # try
# # 创建新特征
# # total_rooms、 total_bedrooms、 population、 households 4个属性间相关性很强，对这些变量进行转化
# train_set['rooms_per_household'] = train_set['total_rooms'] / train_set['households']
# train_set['bedrooms_per_room'] = train_set['total_bedrooms'] / train_set['total_rooms']
# train_set['population_per_household'] = train_set['population'] / train_set['households']
# # 查看新特征与目标变量的相关性
# corr_matrix = corr_matrix_vis(train_set)
# attr_adder = CombineAttributesAdder(add_berooms_per_room=False)
# # pandas.DataFrame.values Return a Numpy representation of the DataFrame
# data_extra_attribs = attr_adder.transform(data.values)
# print('After combine attributes adding:')
# print(data_extra_attribs)

# 数据预处理
train_data, train_labels, test_data, test_labels = get_data_labels(train_set, test_set)
train_data = SuperPipeline(train_data)
test_data = SuperPipeline(test_data)

# 模型建立
# 1.线性回归模型
lin_reg = LinearRegression()
# 训练模型
lin_reg.fit(train_data, train_labels)
train_predictions = lin_reg.predict(train_data)
# 使用 mean_square_error 来测量模型拟合度
lin_mse = mean_squared_error(train_labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)
# 计算测试集的均方误差
test_predictions = lin_reg.predict(test_data)
test_lin_mse = mean_squared_error(test_labels, test_predictions)
test_lin_rmse = np.sqrt(test_lin_mse)

# 2.决策树模型
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_data, train_labels)
train_predictions = tree_reg.predict(train_data)
tree_mse = mean_squared_error(train_labels, train_predictions)
tree_rmse = np.sqrt(tree_mse)
test_predictions = tree_reg.predict(test_data)
test_tree_mse = mean_squared_error(test_labels, test_predictions)
test_tree_rmse = np.sqrt(test_tree_mse)

# 解决决策树过拟合问题
# 措施：限制数的深度or每个节点最小分配数量来抑制过拟合问题
train_mse = []
test_mse = []
parameter_values = range(3, 20)
for i in parameter_values:
    tree_reg = DecisionTreeRegressor(max_depth=i)
    tree_reg.fit(train_data, train_labels)
    train_predictions = tree_reg.predict(train_data)
    tree_mse = mean_squared_error(train_labels, train_predictions)
    tree_rmse = np.sqrt(tree_mse)
    train_mse.append(tree_mse)
    test_predictions = tree_reg.predict(test_data)
    test_tree_mse = mean_squared_error(test_labels, test_predictions)
    test_tree_rmse = np.sqrt(test_tree_mse)
    test_mse.append(test_tree_mse)

plt.figure() 
plt.plot(parameter_values, train_mse, '-o', test_mse, '--o')
plt.show()

train_mse = []
test_mse = []
parameter_values = range(3, 20)
for i in parameter_values:
    tree_reg = DecisionTreeRegressor(min_samples_leaf=i)
    tree_reg.fit(train_data, train_labels)
    train_predictions = tree_reg.predict(train_data)
    tree_mse = mean_squared_error(train_labels, train_predictions)
    tree_rmse = np.sqrt(tree_mse)
    train_mse.append(tree_mse)
    test_predictions = tree_reg.predict(test_data)
    test_tree_mse = mean_squared_error(test_labels, test_predictions)
    test_tree_rmse = np.sqrt(test_tree_mse)
    test_mse.append(test_tree_mse)

plt.figure() 
plt.plot(parameter_values, train_mse, '-o', test_mse, '--o')
plt.show()

# 训练决策树模型
tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=15)
tree_reg.fit(train_data, train_labels)
train_predictions = tree_reg.predict(train_data)
tree_mse = mean_squared_error(train_labels, train_predictions)
tree_rmse = np.sqrt(tree_mse)
test_predictions = tree_reg.predict(test_data)
test_tree_mse = mean_squared_error(test_labels, test_predictions)
test_tree_rmse = np.sqrt(test_tree_mse)

# 3.随机森林
forest_reg = RandomForestRegressor(n_estimators=20, max_depth=5)
forest_reg.fit(train_data, train_labels)
train_predictions = forest_reg.predict(train_data)
forest_mse = mean_squared_error(train_labels, train_predictions)
forest_rmse = np.sqrt(forest_mse)
test_predictions = forest_reg.predict(test_data)
test_forest_mse = forest_reg.predict(test_data)
test_forest_rmse = np.sqrt(test_forest_mse)

# 模型泛化能力对比
# 交叉验证法
# sklearn 的交叉验证倾向使用效用函数（值越大越好），而不是成本函数，这里使用负的MSE
scores = cross_val_score(lin_reg, 
                         train_data, 
                         train_labels, 
                         scoring='neg_mean_squared_error', 
                         cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Std:', scores.std())
    
print('Linear regression:')
display_scores(rmse_scores)
    
scores = cross_val_score(tree_reg, 
                         train_data, 
                         train_labels, 
                         scoring='neg_mean_squared_error',
                         cv=10)
rmse_scores = np.sqrt(-scores)

print('Decision tree regressor:')
display_scores(rmse_scores)

scores = cross_val_score(forest_reg,
                         train_data,
                         train_labels,
                         scoring='neg_mean_squared_error',
                         cv=10)
rmse_scores = np.sqrt(-scores)

print('Random forest regressor:')
display_scores(rmse_scores)