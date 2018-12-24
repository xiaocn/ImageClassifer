#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要定义一些共有的数据特征，文件名等
作者： xiao
时间： 2018.12.23
"""

# 导入future模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入系统模块
import tensorflow as tf
from tensorflow.contrib import slim
import os


def items_to_handlers():
    """
    数据集需要处理的选项，一般为图像数据及标签
    :return: 返回数据集处理的选项
    """
    items_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    return items_handlers


def keys_to_feature():
    """
    record文件的关键子解码器
    :return: 返回选项解码器
    """
    keys_feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    return keys_feature


def image_to_example(image_data, image_format, height, width, class_id):
    """
    图像编码器
    :param image_data: 输入的图像数组
    :param image_format: 图像的编码格式
    :param height: 图像的高
    :param width: 图像的宽
    :param class_id: 该图像对应的标签
    :return: 返回图像的编码序列
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def get_split_name(test_dp=0.1, val_dp=0.1):
    """
    获取数据集的划分列表
    :param test_dp: 测试集占整个数据集的比例，默认值为0.1
    :param val_dp: 验证集占训练集的比例，默认值为0.1
    :return: 返回一个所有划分集名称的列表
    """
    split_name_dict = {
        'test': test_dp,
        'val': (1 - test_dp) * val_dp,
        'train': (1 - test_dp) * (1 - val_dp),
    }
    return split_name_dict


def get_file_name(record_dir, split_name=None, num_shard=0, shard_id=0, file_pattern=None):
    """
    获取各个文件的全路径，包括record文件，标签文件，数据集说明文件
    :param record_dir: record数据集所在的文件夹
    :param split_name: 数据集名称，一般为test，validation，train这三个，默认为None
    :param shard_id: 该数据集的record文件数量的下标
    :param num_shard: 该数据集的record文件总数，当默认为0时，表示返回该数据集所有的record文件
    :param file_pattern: 文件匹配的正则表达式或具体文件名
    :return: 根据情况返回各个文件的全路径，当split_name==None时，返回标签文件和数据集数量说明文件，
                                        当split_name!=None and num_shard>0 时，返回record具体文件名，
                                        否则返回record的模糊文件名,及标签文件名，数据集数量说明文件名
    """
    if num_shard > 0 and split_name is not None:
        return os.path.join(record_dir, '%s_%05d-of-%05d.record' % (split_name, shard_id, num_shard))
    label_file = os.path.join(record_dir, 'labels.txt')
    num_file = os.path.join(record_dir, 'num.txt')
    if split_name is None:
        return label_file, num_file
    if file_pattern is None:
        file_pattern = '%s_*.record' % split_name
    return os.path.join(record_dir, file_pattern), label_file, num_file


def read_dict_data(filename):
    """
    读取文本文件
    :param filename: 文件名（含路径）
    :return: 返回以英文冒号作为分割符划分每行内容的字典数据
    """
    with tf.gfile.Open(filename, 'rb') as f:
        lines = f.read().decode()
    lines = filter(None, lines.split('\n'))
    data = {}
    for line in lines:
        index = line.index(':')
        data[line[:index]] = int(line[index + 1:])
    return data


def create_dict_file(data_dict, filename):
    """
    创建文本文件，创建内容为字典格式，字符串：数字的储存内容
    :param data_dict: 字典数据
    :param filename: 储存的文件名
    :return: 无返回值
    """
    with tf.gfile.Open(filename, 'w') as f:
        key_list = sorted(data_dict)
        for key in key_list:
            value = data_dict[key]
            f.write('%s:%d\n' % (key, value))


def int64_feature(values):
    """
    64位整数列表编码器
    :param values: 64位的整数序列
    :return: 返回64位整数列表的编码序列
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """
    字节编码器
    :param values: 字节序列
    :return: 返回字节列表的编码序列
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """
    浮点数编码器
    :param values: 浮点数序列
    :return: 返回浮点数的编码序列
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))
