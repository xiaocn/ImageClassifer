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


def _get_tfrecords_dir_list(dataset_dir):
    tfrecords_dir_list = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path) and filename.split('_')[-1] == 'tfrecord':
            tfrecords_dir_list.append(path)
    return tfrecords_dir_list


def get_tfrecords_list(dataset_dir):
    tfrecords_train = []
    tfrecords_validation = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if not os.path.isdir(path) and filename.split('.')[-1] == 'tfrecord':
            if filename.split('_')[1] == 'train':
                tfrecords_train.append(path)
            else:
                tfrecords_validation.append(path)


def get_tfrecords_map(dataset_dir):
    tfrecords_dir_list = _get_tfrecords_dir_list(dataset_dir)
    tfrecords_map_train = {}
    tfrecords_map_validation = {}
    for tfrecords_dir in tfrecords_dir_list:
        class_name = tfrecords_dir.split('/')[-1].split('_')[0]
        tfrecords_map_train[class_name] = []
        tfrecords_map_validation[class_name] = []
        for filename in os.listdir(tfrecords_dir):
            filename_split = filename.split('_')[1]
            if filename_split == 'train':
                tfrecords_map_train[class_name].append(os.path.join(tfrecords_dir, filename))
            else:
                tfrecords_map_validation[class_name].append(os.path.join(tfrecords_dir, filename))
    return tfrecords_map_train, tfrecords_map_validation


def read_and_decode(tfrecords_list, batch_size, num_epochs=None, is_training=False):
    filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=num_epochs)
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


def read_and_decode_with_path(tfrecords_list, batch_size, num_epochs=None, is_training=False):
    filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    '''
    tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/path': bytes_feature(path)
    }))
    '''
    image_features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/g': tf.FixedLenFeature([1], tf.int64),
        'image/t': tf.FixedLenFeature([1], tf.int64),
        'image/c': tf.FixedLenFeature([1], tf.int64)
    })
    image_encoded, image_format = image_features['image/encoded'], image_features['image/format']
    image_label = image_features['image/class/label']
    image_g, image_t, image_c = image_features['image/g'], image_features['image/t'], image_features['image/c']
    image_path = tf.concat([image_g, image_t, image_c], axis=0)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)

    image_decoded = image_preprocessing.preprocess(image_decoded, size, size, is_training=is_training)

    image_batch, label_batch, path_batch = tf.train.batch([image_decoded, image_label, image_path],
                                                          batch_size=batch_size, num_threads=4, capacity=5*batch_size,)
    label_batch = tf.one_hot(label_batch, depth=NUM_CLASSES, dtype=tf.float32)
    return image_batch, label_batch, path_batch


def read_decode_mux_with_path(tfrecords_map, batch_size, is_training=True):
    if batch_size % len(tfrecords_map.keys()) != 0:
        return
    batch_size_single_class = int(batch_size / len(tfrecords_map.keys()))
    image_batch_list = []
    label_batch_list = []
    path_batch_list = []
    for class_name in tfrecords_map.keys():
        image_batch, label_batch, path_batch = read_and_decode_with_path(tfrecords_map[class_name],
                                                                         batch_size_single_class,
                                                                         is_training=is_training)
        image_batch_list.append(image_batch)
        label_batch_list.append(label_batch)
        path_batch_list.append(path_batch)
    print(image_batch_list)
    print(label_batch_list)
    print(path_batch_list)
    image_batch_total = image_batch_list[0]
    label_batch_total = label_batch_list[0]
    path_batch_total = path_batch_list[0]
    for index in range(len(image_batch_list)):
        if index == 0:
            continue
        image_batch_total = tf.concat([image_batch_total, image_batch_list[index]], axis=0)
        label_batch_total = tf.concat([label_batch_total, label_batch_list[index]], axis=0)
        path_batch_total = tf.concat([path_batch_total, path_batch_list[index]], axis=0)
    return image_batch_total, label_batch_total, path_batch_total


def read_and_decode_mux(tfrecords_map, batch_size, is_training=True):
    if batch_size % len(tfrecords_map.keys()) != 0:
        return
    batch_size_single_class = int(batch_size / len(tfrecords_map.keys()))
    image_batch_list = []
    label_batch_list = []

    for class_name in tfrecords_map.keys():
        image_batch, label_batch = read_and_decode(tfrecords_map[class_name], batch_size_single_class,
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
