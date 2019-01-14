#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块功能： 实现将record数据集转换为图像
作者： xiao
时间： 2019.1.3
"""

# 导入future模块
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 导入系统模块
import tensorflow as tf
import image_preprocessing
import os

NUM_CLASSES = 2
size = 224


def _get_records_dir_list(dataset_dir):
    records_dir_list = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path) and filename.split('_')[-1] == 'record':
            records_dir_list.append(path)
    return records_dir_list


def get_records_map(dataset_dir):
    records_dir_list = _get_records_dir_list(dataset_dir)
    records_map_train = {}
    records_map_validation = {}
    for records_dir in records_dir_list:
        class_name = records_dir.split('/')[-1].split('_')[0]
        records_map_train[class_name] = []
        records_map_validation[class_name] = []
        for filename in os.listdir(records_dir):
            filename_split = filename.split('_')[1]
            if filename_split == 'train':
                records_map_train[class_name].append(os.path.join(records_dir, filename))
            else:
                records_map_validation[class_name].append(os.path.join(records_dir, filename))
    return records_map_train, records_map_validation


def read_and_decode(records_list, batch_size, num_epochs=None, is_training=False):
    filename_queue = tf.train.string_input_producer(records_list, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    '''
    tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width)
    }))
    '''
    image_features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64)
    })
    image_encode, image_format = image_features['image/encoded'], image_features['image/format']
    image_label = image_features['image/class/label']
    image_decoded = tf.image.decode_jpeg(image_encode, channels=3)
    image_decoded = image_preprocessing.preprocess(image_decoded, size, size, is_training=is_training)
    image_batch, label_batch = tf.train.batch([image_decoded, image_label], batch_size=batch_size, num_threads=4,
                                              capacity=5*batch_size,)
    label_batch = tf.one_hot(label_batch, depth=NUM_CLASSES, dtype=tf.float32)
    return image_batch, label_batch


def read_and_decode_mux(records_map, batch_size, is_training=True):
    if batch_size % len(records_map.keys()) != 0:
        return
    batch_size_single_class = int(batch_size / len(records_map.keys()))
    image_batch_list = []
    label_batch_list = []

    for class_name in records_map.keys():
        image_batch, label_batch = read_and_decode(records_map[class_name], batch_size_single_class,
                                                   is_training=is_training)
        image_batch_list.append(image_batch)
        label_batch_list.append(label_batch)
    image_batch_total, label_batch_total = image_batch_list[0], label_batch_list[0]
    for index in range(len(image_batch_list)):
        if index == 0:
            continue
        image_batch_total = tf.concat([image_batch_total, image_batch_list[index]], axis=0)
        label_batch_total = tf.concat([label_batch_total, label_batch_list[index]], axis=0)
    return image_batch_total, label_batch_total
