# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:11:45 2020

@author: Administrator
"""


import numpy as np
from sklearn.model_selection import train_test_split
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# 方式1.基于随机数划分训练集和测试集
def split_train_test(data, test_ratio, random_state=42):
    '''
        生成训练集和测试集
    '''
    # numpy.random.seed() Reseed a legacy MT19937 BigGenerator.
    # 设置随机种子，使随机过程可复现
    np.random.seed(random_state)
    # numpy.random.permutation() Randomly permute a sequence, or return a permuted range.
    # 生成不重复的下标随机数
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    # 测试集的下标集合
    test_indicies = shuffled_indicies[:test_set_size]
    # 训练集的下标集合
    train_indicies = shuffled_indicies[test_set_size:]
    # pandas.DataFrame.iloc[] Purely integer-location based indexing for selection by position.
    train_set, test_set = data.iloc[train_indicies], data.iloc[test_indicies]
    # 打印训练集和测试集的规模
    print('After randomly splitting:')
    print('Train set scale is {}'.format(len(train_set)))
    print('Test set scale is {}'.format(len(test_set)))
    
    return train_set, test_set

# 随机划分训练集和测试集
# flag: local
# if local == False: use method in sklearn
# if local == True: use method in local module
# 实际操作常调用sklearn函数
def randomly_split(data, test_size, random_state=42, local=False):
    if local:
        return split_train_test(data, test_ratio=test_size, random_state=random_state)
    else:
        # sklearn.model_selection.train_test_split() Split arrays or matrices into random train and test subsets.
        train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
        # 打印训练集和测试集的规模
        print('After randomly splitting:')
        print('Train set scale is {}'.format(len(train_set)))
        print('Test scale is  {}'.format(len(test_set)))
        return train_set, test_set
    
# 方式2.基于唯一标识划分训练集与测试集
# 选择测试集，新增数据亦可划分
def test_set_check(identifier, test_ratio, hash):
    # return boolean value
    # numpy.int64 Signed integer type, compatible with C long long.
    # hash() Return the hash value for the given object. Two obejcts that compare equl must have the same hash value, but the reverse is not necessarily true.
    # hash() --> hashlib.md5() defaultly here
    # 取对唯一标识进行哈希映射的结果中最后一个字节，与目标条件比较
    return hash(np.int64(identifier)).digest()[-1] < (256 * test_ratio)

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    # 取得用于进行哈希映射的唯一标识Series
    ids = data[id_column]
    # 检查唯一标识是否在测试集中
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    train_set, test_set = data.loc[~in_test_set], data.loc[in_test_set]
    # 打印训练集和测试集的规模
    print('After hash(id) splitting:')
    print('Train set scale is {}'.format(len(train_set)))
    print('Test scale is  {}'.format(len(test_set)))
    return train_set, test_set

def hash_id_split(data, test_size, id_column=None, hash=hashlib.md5):
    # 若未指定唯一标识所在Series，则将数据集的Index转为一列，并重建其Index
    if not id_column:
        # pandas.DataFrame.reset_index() Reset the index, or a level of it.
        data_with_id = data.reset_index()
        train_set, test_set = split_train_test_by_id(data_with_id, test_ratio=test_size, id_column='index', hash=hash)
    else:
        train_set, test_set = split_train_test_by_id(data, test_ratio=test_size, id_column=id_column, hash=hash)
    return train_set, test_set

# 方式3.分层随机选取
# 经特征工程发现：
# median_income 属性对 median_house_value 影响十分巨大，这个变量大约可以解释70%的房价
# 为避免模型的评估出现偏差，我们在选取数据的时候要保证数据集中的每个分层都有足够的实例位于我们的训练集中
def stratified_split(data, test_size, random_state=42, stratified_column=None):
    # stratified_column 未被使用，后续优化代码
    # 离散化
    # 缩放+取整
    # numpy.ceil() Return the ceilling of the input, element-wise.
    data['income_category'] = np.ceil(data['median_income'] / 1.5)
    # 长尾分布收缩
    # pandas.DataFrame.where() Replace values where the condition is False.
    data['income_category'].where(data['income_category'] < 5, 5.0, inplace=True)
    plt.figure()
    data['income_category'].hist(bins=5, histtype='stepfilled')
    plt.title('median_income --> income_category')
    plt.show()
    # 查看每个类别在总数中所占比例
    proportion_series = data['income_category'].value_counts() / data['income_category'].size
    proportion_series = proportion_series.apply(lambda x: '{:.2f}%'.format(x * 100)).sort_index()
    print('The proportion of unique values in income_category:')
    print(proportion_series)
    # 分层随机抽取测试集
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    result = split.split(data, data['income_category'])
    for train_index, test_index in result:
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    # 打印训练集和测试集的规模
    print('After stratified splitting:')
    print('Train set scale is {}'.format(len(strat_train_set)))
    print('Test scale is  {}'.format(len(strat_test_set)))    
    # 训练集中每个类别的数据比例
    proportion_series = strat_train_set['income_category'].value_counts() / strat_train_set['income_category'].size
    proportion_series = proportion_series.apply(lambda x: '{:.2f}%'.format(x * 100)).sort_index()
    print('The proportion of unique values in income_category of stratified train set:')
    print(proportion_series)
    # 删除 income_category 列
    for set in (strat_train_set, strat_test_set):
        set.drop(['income_category'], axis=1, inplace=True)
    
    return strat_train_set, strat_test_set