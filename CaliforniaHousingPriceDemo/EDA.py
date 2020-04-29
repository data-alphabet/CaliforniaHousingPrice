# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:20:42 2020

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 打印用于区域开始处的分割线
def print_start_separator(separator='*', line_len=50):
    print(separator * line_len)

# 打印用于区域结束处的分割线    
def print_end_separator(separator='*', line_len=50):
    print(separator * line_len + '\n')

# 对数据集进行探索性数据分析
# 1.随机抽样
# 2.简要总结（数据集规模、列名、非空值计数、数据类型）
# 3.字符串型变量构成比例分析
# 4.数值型变量描述性统计（计数、均值、标准差、百分位数）
# 5.绘制直方图，快速了解数据分布状况
def exploratory_data_analysis(data, 
                              max_columns=20, 
                              random_sample_size=5, 
                              text_columns=['ocean_proximity'], 
                              hist_bins=50, 
                              figsize=(20, 15)):
    # 设置最大显示列数为20，以在控制台显示完整的列数据
    pd.set_option('display.max_columns', max_columns)
    
    # pandas.DataFrame.sample() Return a random sample of items from an axis of object.
    # You can use random_state for reproducibility.
    # 从 data 中随机抽取5个对象组成一个随机抽样样本
    data_sample = data.sample(random_sample_size)
    print('Random sampling:')
    print_start_separator()
    print(data_sample)
    print_end_separator()
    
    # pandas.DataFrame.info() Print a concise summary of a DataFrame.
    # 打印 data 的简要总结
    print('A concise summary of dataset:')
    print_start_separator()
    data.info()
    print_end_separator()
    
    # # 数据集只有一个是字符串型的类别变量
    # # pandas.Series.value_counts() Return a Series containing couts of unique values.
    # # 查看变量中各唯一值的数量
    # print('The counts of unique values in "ocean_proximity":')
    # print_start_separator()
    # print(data['ocean_proximity'].value_counts())
    # print_end_separator()
    
    # 字符串型的类别变量唯一值的占比统计
    for text_column in text_columns:
        # 取得字符型变量的Series
        text_series = data[text_column]
        
        # pandas.Series.size Return the number of elements in the underlying data.
        # 取得Series的长度（值的个数）
        # 计算变量中唯一值的占比
        proportion_series = text_series.value_counts() / text_series.size
        
        # pandas.Series.apply() Invoke function on values of Series.
        # lambda 表达式
        # 1.语法：
        # lambda argument_list: expression
        # 2.特性：
        # a) lambda 函数是匿名的
        # b) lambda 函数有输入和输出
        # c) lambda 函数一般功能简单
        # 3.用法：
        # a) 将 lambda 函数赋值给一个变量，通过这个变量间接调用该 lambda 函数
        #   add = lambda x, y: x + y
        #   add(1, 2) --> return 3
        # b) 将 lambda 函数赋值给其他函数，从而将其他函数用该 lambda 函数替换
        #   time.sleep = lambda x: None
        #   time.sleep(3)
        # c) 将 lambda 函数作为其他函数的返回值，返回给调用者（嵌套函数、内部函数） 闭包（Closure）
        #   return lambda x, y: x + y
        # d) 将 lambda 函数做为参数传递给其他函数
        #   典型内置函数：
        #   filter: filter(lambda x: x % 3 == 0, [1, 2, 3])
        #   sorted: sorted([1,2,3,4,5,6,7,8,9], key=lambda x: abs(5 - x))
        #   map: map(lambda x: x + 1, [1,2,3])
        #   reduce: reduce(lambda a, b: '{},{}'.format(a, b), [1,2,3,4,5,6,7,8,9])
        # 4.争议：
        # 支持:使用 lambda 编写的代码更紧凑，更“pythonic”
        # 反对：lambda 函数能够支持的功能十分有限，不支持多分支if...elif...else和
        # 异常处理try...except... 并且 lambda 函数的功能被隐藏，理解代码需要耗费一定的理解成本
        # 将占比数值格式化为百分比显示
        # format() Return value.__format__(format_spec)
        proportion_series = proportion_series.apply(lambda x: '{:.2f}%'.format(x * 100))
        
        print('The proportion of unique values in {}:'.format(text_column))
        print_start_separator()
        print(proportion_series)
        print_end_separator()
    
    # pandas.DataFrame.describe() Generate descriptive statistics.
    # 数据集描述性统计
    print('Descriptive statistics of dataset:')
    print_start_separator()
    print(data.describe())
    print_end_separator()
    
    # 绘制每个属性的直方图，快速了解数据分布状况
    # Make a histogram of the DataFrame which is a representation of the distribution of data.
    # This function calls matplotlib.pyplot.hist(), on each Series in the DataFrame, resulting
    # in one histogram per column.
    # bins: int or sequence, default 10. Number of histogram bins to be used.
    # figsize: tuple. The size in inches of the figure to create.
    data.hist(bins=hist_bins, figsize=figsize)
    # matplotlib.pyplot.show() Display a figure.
    # 显示图片
    plt.show()
    
    # 保存图片
    # 图表存放目录
    CHARTS_PATH = './Charts'
    # 如果目录不存在，创建目录
    if not os.path.isdir(CHARTS_PATH):
        # os.mkdir() Create a directory.
        # 创建图表的存放目录
        os.mkdir(CHARTS_PATH)
    fig_name = 'Histogram.png'
    # 图片存放路径
    fig_path = os.path.join(CHARTS_PATH, fig_name)
    # matplotlib.pyplot.savefig() Save the current figure.
    # 保存当前图片
    plt.savefig(fig_path,
                # format='png'
                )


if __name__ == '__main__':
    # 载入数据
    # pandas.read_csv() Read a comma-separated value(csv) file into DataFrame.
    # data 是一个 DataFrame
    data = pd.read_csv('./datasets/housing/housing.csv')
    
    exploratory_data_analysis(data)