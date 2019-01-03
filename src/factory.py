#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要实现数据集的读取与写入record文件
作者： xiao
时间： 2018.12.23
"""

# 导入future模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入自定义模块
from dataset import folder_dataset as folder
from dataset import mnist_dataset as mnist


dataset_map = {
    'flower': folder,
    'mnist': mnist,
}


def get_dataset(dataset_name):
    """
    获取具体数据集模块
    :param dataset_name: 数据集的名称
    :return: 返回具体数据集模块
    """
    assert dataset_name in ['flower', 'mnist']
    return dataset_map[dataset_name]
