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


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(image_batch, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([784, 500], regularizer)
        biases = tf.get_variable('biases', [500], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(image_batch, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([500, 10], regularizer)
        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2