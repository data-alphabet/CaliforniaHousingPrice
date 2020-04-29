# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:29:27 2020

@author: Administrator
"""


import os
import tarfile
from six.moves import urllib

# 可能报错：urlopen eroor [Errno 11004] getaddrinfo failed
# 原因：Github的raw文件读取地址遭污染
# 解决方案：
# step1.在IPAddress.com https://www.ipaddress.com 上搜索 raw.githubusercontent.com
#       得到domain name对应的IP：199.232.68.133
# step2.修改Windows系统的hosts文件，增添一行
#       199.232.68.133 raw.githubusercontent.com

# 数据文件下载的根目录
DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
# os.path.join() join two (or more) paths.
# HOUSING_PATH: 'datasets\\housing'
# 数据文件在本地的存放目录
HOUSING_PATH = os.path.join('datasets','housing')
# 数据文件下载的URL
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # os.path.isdir() Return true if the pathname refers to an existing directory.
    # 如果目录不存在，创建目录
    if not os.path.isdir(housing_path):
        # os.makeidrs() Super-mkdir; Works like mkdir
        # 创建数据文件在本地的存放目录
        os.makedirs(housing_path)
    # tgz文件在本地的路径
    tgz_path = os.path.join(housing_path, 'housing_tgz')
    # urllib.request.urlretrieve() Retrieve a URL into a temporary location on disk.
    # 检索URL加载到本地硬盘上指定的位置
    urllib.request.urlretrieve(housing_url, tgz_path)
    # tarfile.open() Open a tar archive for reading, writing or appending.Return an appropriate TarFile class.
    # 打开tar文档，mode='r'
    housing_tgz = tarfile.open(tgz_path)
    # tarfile.TarFile.extractall() Extract all members from the archive to the current working directory ...
    # 解压缩tar包到指定路径
    housing_tgz.extractall(path=housing_path)
    # tarfile.TarFile.close() Close the TarFile.
    # 关闭tar文档
    housing_tgz.close()

if __name__ == '__main__':
    fetch_data()