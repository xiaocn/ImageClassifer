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
import time

import input_data
import mobilenet_v1

import tensorflow.contrib.slim as slim
import tools

batch_size = 1
num_classes = 2


def network_fn(images, num_classes, arg_scope, func, is_training=True):
    with slim.arg_scope(arg_scope):
        return func(images, num_classes, is_training=is_training)


def evaluate(sess, predict, accuracy, images, labels, image_batch, label_batch, is_muddy=True):
    i = 1
    num_true = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            image_batch_value, label_batch_value = sess.run([image_batch, label_batch])

            predict_value, accuracy_value = sess.run([predict, accuracy], feed_dict={
                images: image_batch_value,
                labels: label_batch_value})
            num_true += accuracy_value
            i += 1
            break
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        print('num of examples:', i)
        if is_muddy:
            print('muddy_accuracy:', num_true / (i - 1) * 100, '%')
        else:
            print('clear_accuracy:', num_true / (i - 1) * 100, '%')
        coord.request_stop()

    coord.join(threads)
    return i - 1, num_true


def run(model_save_path, dataset_dir):
    with tf.Graph().as_default():
        records_map_train, records_map_validation = input_data.get_records_map(dataset_dir)

        muddy_image_batch, muddy_label_batch = input_data.read_and_decode(
            records_map_validation['muddy'], batch_size, num_epochs=1, is_training=False)
        clear_image_batch, clear_label_batch = input_data.read_and_decode(
            records_map_validation['clear'], batch_size, num_epochs=1, is_training=False)

        images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
        labels = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)

        logits, endpoints = network_fn(images, num_classes, arg_scope, mobilenet_v1.mobilenet_v1, is_training=False)

        predict = endpoints['Predictions']
        accuracy = tools.accuracy(logits=predict, labels=labels)

        saver = tf.train.Saver(slim.get_model_variables())

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            saver.restore(sess, model_save_path)
            print('muddy:')
            num_muddy, num_true_muddy = evaluate(sess, predict, accuracy, images, labels,
                                                 muddy_image_batch, muddy_label_batch)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            saver.restore(sess, model_save_path)
            print('clear:')
            num_clear, num_true_clear = evaluate(sess, predict, accuracy, images, labels,
                                                 clear_image_batch, clear_label_batch, is_muddy=False)
        global_accuracy = (num_true_muddy + num_true_clear) / (num_muddy + num_clear)
        print('global_accuracy:', (num_true_muddy + num_true_clear) / (num_muddy + num_clear) * 100, '%')

    num_true_positive = num_true_clear
    num_false_positive = num_clear - num_true_clear
    num_false_negative = num_muddy - num_true_muddy
    precision = num_true_positive / (num_true_positive + num_false_positive)
    recall = num_true_positive / (num_true_positive + num_false_negative)
    f1 = 2. / (1. / precision + 1. / recall)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    return global_accuracy, precision, recall, f1


def run_k_fold(model_step):
    global_accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    k = 6
    for kflod in range(k):
        print("kfold:", kflod)
        dataset_dir = './small_dataset/k_fold_' + str(kflod)
        global_accuracy, precision, recall, f1 = run(
            'small/models_pro_k_fold/k_fold_' + str(kflod) + '/model.ckpt-' + str(model_step), dataset_dir)
        global_accuracy_mean += global_accuracy
        precision_mean += precision
        recall_mean += recall
        f1_mean += f1
        print()
    global_accuracy_mean /= k
    precision_mean /= k
    recall_mean /= k
    f1_mean /= k
    print('global_accuracy_mean:', global_accuracy_mean)
    print('precision_mean:', precision_mean)
    print('recall_mean:', recall_mean)
    print('f1_mean:', f1_mean)
    return global_accuracy_mean, precision_mean, recall_mean, f1_mean


if __name__ == '__main__':
    max_accuracy = (0, 0)
    max_precision = (0, 0)
    max_recall = (0, 0)
    max_f1 = (0, 0)
    for num in range(2100, 5100, 100):
        print('train_steps', num, ':')
        global_accuracy, precision, recall, f1 = run_k_fold(num)
        if global_accuracy > max_accuracy[0]:
            max_accuracy = (global_accuracy, num)
        if precision > max_precision[0]:
            max_precision = (precision, num)
        if recall > max_recall[0]:
            max_recall = (recall, num)
        if f1 > max_f1[0]:
            max_f1 = (f1, num)
        print()
    print('max_accuracy:', max_accuracy[0], 'step:', max_accuracy[1])
    print('max_precision:', max_precision[0], 'step:', max_precision[1])
    print('max_recall:', max_recall[0], 'step:', max_recall[1])
    print('max_f1:', max_f1[0], 'step:', max_f1[1])

    # global_accuracy, precision, recall, f1 = run_k_fold(900)
