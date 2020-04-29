# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:30:07 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# 地理数据可视化
def geo_data_vis(data, x='longitude', y='latitude', s=None, label=None, c=None):
    if not s:
        s = data['population'] / 100
    if not label:
        label = 'population'
    if not c:
        c = 'median_house_value'
    # 散点图
    data.plot(kind='scatter',
              x=x,# 横坐标为经度
              y=y,# 纵坐标为纬度
              s=s,# the marker size in points ** 2
              label=label,# 
              c=c,# marker color
              cmap=plt.get_cmap('jet'),# Get a colormap instance, defaulting to rc value if name is None.
              colorbar=True,# 显示颜色条/渐变色条
              alpha=0.4# 调和值，即透明度/不透明度
              )
    plt.legend()
    plt.show()
    # 保存图片
    # 图表存放目录
    CHARTS_PATH = './Charts'
    # 如果目录不存在，创建目录
    if not os.path.isdir(CHARTS_PATH):
        # os.mkdir() Create a directory.
        # 创建图表的存放目录
        os.mkdir(CHARTS_PATH)
    fig_name = 'Geographic.png'
    # 图片存放路径
    fig_path = os.path.join(CHARTS_PATH, fig_name)
    # matplotlib.pyplot.savefig() Save the current figure.
    # 保存当前图片
    plt.savefig(fig_path,
                # format='png'
                )
    
# 相关性矩阵计算及可视化
def corr_matrix_vis(data, figsize=(15, 12), target_column='median_house_value'):
    corr_matrix = data.corr()
    # 打印其他变量与目标变量的相关系数（降序）
    print('The correlation coefficients of other attributes with {} are as follow:'.format(target_column))
    print(corr_matrix[target_column].sort_values(ascending=False))
    
    
    plt.figure(figsize=figsize)
    # 热力图
    # seaborn.heatmap() Plot rectangular data as a color-encoded matrix.
    sns.heatmap(corr_matrix,
                annot=True,# If True, write the data value in each cell.
                cmap='RdBu_r',# The mapping from data values to color space.
                linewidth=0.1,# Width of the lines that will divide each cell.
                linecolor='white',# Color of the lines that divide each cell.
                vmax=0.9,# Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
                square=True# If True, set the Axes aspect to "equal" so each cell will be square-shaped.
                )
    plt.title('Correlations Among Features', # label: Text to use for the title.
              # y=1.03, # no-used parameter
              fontsize=17# 字体大小
              )
    plt.show()
    # 保存图片
    # 图表存放目录
    CHARTS_PATH = './Charts'
    # 如果目录不存在，创建目录
    if not os.path.isdir(CHARTS_PATH):
        # os.mkdir() Create a directory.
        # 创建图表的存放目录
        os.mkdir(CHARTS_PATH)
    fig_name = 'CorrelationMatrix.png'
    # 图片存放路径
    fig_path = os.path.join(CHARTS_PATH, fig_name)
    # matplotlib.pyplot.savefig() Save the current figure.
    # 保存当前图片
    plt.savefig(fig_path,
                # format='png'
                )
    return corr_matrix


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# 属性组合添加类
# sklearn.base.BaseEstimator Base classes for all estimators.估计器
# sklearn.base.TransformerMixin Mixin class for all transformers in scikit-learn.转换器
class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_berooms_per_room=True):
        self.add_berooms_per_room = add_berooms_per_room
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X, Y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_berooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            # 将数据按列拼接
            # numpy.c_[] Translate slice objects to concatenation along the second axis.
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]