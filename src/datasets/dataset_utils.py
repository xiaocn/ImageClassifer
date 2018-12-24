#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要实现创建数据集或读取数据集的一些工具
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
import sys
import math
import os

# 导入自定义模块
from datasets import dataset_features as features


def dataset_exists(record_dir, filename_dict, num_per_shard, class_names_to_ids):
    """
    判断record数据集是否存在
    :param record_dir: record数据集的根目录
    :param filename_dict: 划分集字典
    :param num_per_shard: 每个record文件所含数据的数量
    :param class_names_to_ids 标签名称下标字典
    :return: True-数据集文件已存在 False-不存在某些数据集文件
    """
    num_shard_dict = {}
    for split_name in filename_dict:
        num_shard_dict[split_name] = len(filename_dict[split_name])
        num_shard = math.ceil(num_shard_dict[split_name] / num_per_shard)
        for shard_id in range(num_shard):
            if not tf.gfile.Exists(
                    features.get_file_name(record_dir, split_name, num_shard=num_shard, shard_id=shard_id)):
                return False
    label_file, num_file = features.get_file_name(record_dir)
    if not tf.gfile.Exists(label_file):
        features.create_dict_file(class_names_to_ids, label_file)
    if not tf.gfile.Exists(num_file):
        features.create_dict_file(num_shard_dict, num_file)
    return True


def get_data_partition(filename_list, test_dp=0.1, val_dp=0.1):
    """
    根据各个划分集比例划分数据集
    :param filename_list: 已随机化的所有数据集的文件列表
    :param test_dp: 测试集占整个数据集的比例
    :param val_dp: 验证集占训练集的比例
    :return: 返回各个已划分好的数据集字典
    """
    assert 0 <= test_dp < 1 and 0 <= val_dp < 1
    dataset_len = len(filename_list)
    filename_dict = {}
    split_name_dict = features.get_split_name(test_dp=test_dp, val_dp=val_dp)
    start = 0
    for split_name in split_name_dict:
        data_num = math.ceil(dataset_len * split_name_dict[split_name])
        filename_dict[split_name] = filename_list[start:(data_num + start)]
        start = data_num
    return filename_dict


def create_dataset(dataset, record_dir, filename_dict, class_names_to_ids, num_per_shard=1000):
    """
    创建数据集
    :param dataset: 具体数据集对象
    :param record_dir: 生成的record文件的保存路径
    :param filename_dict: 划分集字典
    :param class_names_to_ids: 标签字典
    :param num_per_shard: 每个record文件的容纳数据的数量
    :return: 无返回值
    """
    assert num_per_shard > 0
    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)
    num_shard_dict = {}
    for split_name in filename_dict:
        filenames = filename_dict[split_name]
        num_shard_dict[split_name] = len(filenames)
        num_shard = math.ceil(num_shard_dict[split_name] / num_per_shard)
        for shard_id in range(num_shard):
            output_filename = features.get_file_name(record_dir, split_name, shard_id=shard_id, num_shard=num_shard)
            with tf.python_io.TFRecordWriter(output_filename) as record_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> 正在装换%s数据集图像 %d/%d shard %d' %
                                     (split_name, i + 1, len(filenames), shard_id))
                    sys.stdout.flush()
                    image_data, height, width, class_id = dataset.get_image_and_label(filenames, class_names_to_ids, i)
                    example = features.image_to_example(image_data, b'jpg', height, width, class_id)
                    record_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()
    label_file, num_file = features.get_file_name(record_dir)
    features.create_dict_file(num_shard_dict, num_file)
    features.create_dict_file(class_names_to_ids, label_file)
    print("数据集创建完毕！，详情请到目录%s下查看" % record_dir)


def get_dataset(record_dir, split_name, dataset, reader=tf.TFRecordReader):
    """
    读取record数据集
    :param record_dir: record文件所在的路径
    :param split_name: 划分集名称
    :param dataset: 数据集对象
    :param reader: 读数据的对象
    :return: 返回Dataset对象
    """
    record_file, label_file, num_file = features.get_file_name(record_dir, split_name)
    label_name_dict = features.read_dict_data(label_file)
    split_name_dict = features.read_dict_data(num_file)
    item_desc_dict = dataset.get_dataset_desc()
    decoder = slim.tfexample_decoder.TFExampleDecoder(features.keys_feature, features.items_handlers)
    return slim.dataset.Dataset(data_sources=record_file, reader=reader, decoder=decoder,
                                num_samples=split_name_dict[split_name],
                                items_to_descriptions=item_desc_dict,
                                num_classes=len(label_name_dict),
                                labels_to_names=label_name_dict)
