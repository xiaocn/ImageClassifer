#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块功能： 实现将以文件夹为分类标签的数据集转换为tfrecord数据集
用法实例： python3 record_data_test.py --record_dir=<record_dir>
        参数说名： record_dir —— 生成record文件的保存目录
            注： 中括号表示可选参数，等号后为默认值
作者： xiao
时间： 2018.12.21
"""

# 导入future模块
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 导入系统模块
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

# 导入自定义模块
from datasets import dataset_utils

# 定义命令行参数
tf.flags.DEFINE_string('record_dir', None, "生成record文件的保存目录")
tf.flags.DEFINE_string('split_name', None, "划分数据集的名称，一般为train，validation，test三个")
FLAGS = tf.flags.FLAGS


def main(_):
    record_file, label_file, num_file = get_file_name(FLAGS.record_dir, FLAGS.split_name)
    label_name_dict = read_dict_data(label_file)
    split_name_dict = read_dict_data(num_file)
    item_desc_dict = dataset_utils.items_to_desc(split_name_dict)
    keys_to_features = dataset_utils.keys_to_features()
    items_to_handlers = dataset_utils.items_to_handlers()
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=record_file, reader=tf.TFRecordReader, decoder=decoder,
                                   num_samples=split_name_dict[FLAGS.split_name],
                                   items_to_descriptions=item_desc_dict,
                                   num_classes=len(label_name_dict),
                                   labels_to_names=label_name_dict)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * 32,
        common_queue_min=10 * 32)
    [image, label] = provider.get(['image', 'label'])
    images, labels = tf.train.batch(
        [image, label],
        batch_size=32,
        num_threads=4,
        capacity=5 * 32)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes - 0)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2)
    images, labels = batch_queue.dequeue()
    print(label)


if __name__ == '__main__':
    tf.app.run()
