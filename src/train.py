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
from tensorflow.contrib import slim
import os
import input_data
import mobilenet_v1
import tools


batch_size = 64
num_classes = 2
checkpoint_path = ''

learning_rate = 0.001
training_steps = 3000

learning_rate_base = 0.001
decay_steps = 1000 // batch_size
decay_rate = 0.99

MODEL_NAME = 'model.ckpt'

LOG_TRAIN_PATH = 'log/train'
LOG_VAL_PATH = 'log/val'


def network_inference(images, num_classes, arg_scope, func, is_training=True, dropout_keep_prob=0.8):
    with slim.arg_scope(arg_scope):
        return func(images, num_classes, is_training=is_training, dropout_keep_prob=dropout_keep_prob)


def get_variables_to_restore(exclusions):
    variables_to_restore = []
    for var in slim.get_model_variables():
        var_name = var.op.name
        excluded = False
        for exclusion in exclusions:
            if var_name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_variables_to_train(trainable_scopes):
    variables_to_train = []
    for scope in trainable_scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.append(variables)
    return variables_to_train


def train(dataset_dir, model_save_path):
    if not tf.gfile.Exists(model_save_path):
        tf.gfile.MakeDirs(model_save_path)
    with tf.Graph().as_default():
        tfrecords_map_train, tfrecords_map_validation = input_data.get_tfrecords_map(dataset_dir)
        image_batch, label_batch = input_data.read_and_decode_mux(tfrecords_map_train, batch_size)
        val_image_batch, val_label_batch = input_data.read_and_decode_mux(tfrecords_map_validation, batch_size,
                                                                          is_training=False)
        images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
        labels = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=True, weight_decay=0.0004)
        logits, endpoints = network_inference(images, num_classes, arg_scope, mobilenet_v1.mobilenet_v1)
        exclusions = ['MobilenetV1/Logits']
        variables_to_restore = get_variables_to_restore(exclusions)
        cross_entropy = tools.cross_entropy_loss(logits=logits, labels=labels, weights_decay=1.0, name='Logits')
        regularizer_loss = tools.regularizer_loss()
        total_loss = tools.total_loss()

        global_step = tf.Variable(0, trainable=False)

        learning_rate = learning_rate_base

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                    global_step=global_step)
        predict = endpoints['Predictions']
        accuracy = tools.accuracy(logits=predict, labels=labels)

        saver = tf.train.Saver(variables_to_restore)
        saver_last = tf.train.Saver(slim.get_model_variables(), max_to_keep=30)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, checkpoint_path)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
            val_writer = tf.summary.FileWriter(LOG_VAL_PATH, sess.graph)

            i = 1
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                while not coord.should_stop() and i <= training_steps:
                    image_batch_value, label_batch_value = sess.run([image_batch, label_batch])
                    if i == 1 or i % 30 == 0:
                        summary_str = sess.run(summary_op, feed_dict={images: image_batch_value,
                                                                      labels: label_batch_value})
                        train_writer.add_summary(summary_str, i)
                    _, loss_value, accuracy_value = sess.run([train_op, cross_entropy, accuracy],
                                                             feed_dict={images:image_batch_value,
                                                                        labels:label_batch_value})
                    if i == 1 or i % 10 == 0:
                        print(i, 'loss:', loss_value, ', accuracy:', accuracy_value*100, '%')
                    if i == 1 or i % 50 == 0:
                        val_image_batch_value, val_label_batch_value = sess.run([val_image_batch,val_label_batch])
                        loss_value, accuracy_value = sess.run([cross_entropy, accuracy_value], feed_dict={
                            images: val_image_batch_value,
                            labels: val_label_batch_value
                        })
                        print(i, 'val loss:', loss_value, ', accuracy:', accuracy_value*100, '%')
                        summary_str = sess.run(summary_op, feed_dict={images: val_image_batch_value,
                                                                      labels: val_label_batch_value})
                        val_writer.add_summary(summary_str, i)
                    if i % 100 == 0:
                        saver_last.save(sess, os.path.join(model_save_path, MODEL_NAME), global_step=i)
                    i += 1
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)



