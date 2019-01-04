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
import numpy as np


def cross_entropy_loss(logits, labels, weights_decay=1.0, name='cross_entropy'):
    with tf.name_scope(name) as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        loss = tf.multiply(cross_entropy_mean, weights_decay)
        tf.add_to_collection('losses', loss)
        tf.summary.scalar(scope+'/cross_entropy_loss', loss)
        return loss


def regularizer_loss(name='regularizer'):
    with tf.name_scope(name) as scope:
        loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regularizer_loss')
        tf.add_to_collection('losses', loss)
        tf.summary.scalar(scope+'/regularizer_loss', loss)
        return loss


def total_loss(name='total'):
    with tf.name_scope(name) as scope:
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar(scope+'/total_loss', loss)
        return loss


def accuracy(logits, labels, name='accuracy'):
    with tf.name_scope(name) as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar(scope+'accuracy', accuracy_value)
        return accuracy_value


def optimizer(losses, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=losses, global_step=global_step)
        return train_step


def moving_average(decay, global_step):
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        return variable_averages_op


def print_all_variables(sess, train_only=True):
    output = open('variables.npy', 'wb')
    variables = []
    if train_only:
        t_vars = tf.trainable_variables()
        print('[*] printing trainable variables')
    else:
        try:
            t_vars = tf.global_variables()
        except:
            t_vars = tf.all_variables()
        print('[*] printing global variables')
    for idx, v in enumerate(t_vars):
        print(' var {:3}: {:15} {}'.format(idx, str(v.get_shape()), v.name))
        value = sess.run(v)
        print(value)
        print(value.shape)
        print(v.name)
        variables.append(value)
    np.savez(output, fc=variables)
    output.close()