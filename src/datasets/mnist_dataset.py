#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要定义mnist数据集
作者： xiao
时间： 2018.12.23
"""

# 导入future模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入自定义模块
import os
import gzip
import numpy as np
import random


dataset_desc_dict = {  # 描述该数据集的图像标签特性
    'image': "灰度图像，维度为[28 x 28 x 1]",
    'label': "整数下标，范围为[0,9]",
}


def extract_data(filename, data_num, bit_num):
    """
    解压缩.gz文件
    :param filename: 压缩文件名
    :param data_num: 压缩的数据量
    :param bit_num: 以多少位字节解压文件
    :return: 返回解压后的数据
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(bit_num)
        buf = bytestream.read(data_num)
        data = np.frombuffer(buf, dtype=np.uint8)
    return data


def get_dataset_and_labels(dataset_dir):
    """
    读取数据集中的标签及数据的文件路径
    :param dataset_dir: 数据集的根目录，该目录下主要有四个压缩文件
    :return: 返回标签列表及数据路径列表
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'size', 'seven', 'eight', 'nine']
    train_image = extract_data(os.path.join(dataset_dir, 'train-images-idx3-ubyte.gz'), 60000*28*28*1, 16)
    test_image = extract_data(os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz'), 10000*28*28*1, 16)
    data = np.append(train_image.reshape(60000, 28, 28, 1), test_image.reshape(10000, 28, 28, 1), axis=0)
    train_label = extract_data(os.path.join(dataset_dir, 'train-labels-idx1-ubyte.gz'), 1*60000, 8)
    test_label = extract_data(os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz'), 1*10000, 8)
    label = np.append(train_label.astype(np.int64), test_label.astype(np.int64), axis=0)
    photo_list = list(zip(data, label))
    random.seed(0)
    random.shuffle(photo_list)
    return photo_list, dict(zip(class_names, range(len(class_names))))


def get_image_and_label(filenames, class_names_to_ids, index):
    """
    返回单张图像及标签
    :param filenames: 数据文件列表
    :param class_names_to_ids: 标签列表
    :param index: 数据的下标
    :return: 返回第index张图像及标签
    """
    image_data = filenames[index][0].tobytes()
    class_id = filenames[index][1]
    return image_data, 28, 28, class_id
