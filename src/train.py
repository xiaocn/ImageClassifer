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

# 定义参数
tf.flags.DEFINE_integer('batch_size', 64, "批数据大小")
tf.flags.DEFINE_integer('num_classes', 2, "类别数量")
tf.flags.DEFINE_string('checkpoint_path', 'pre_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt', "预训练模型路径")
tf.flags.DEFINE_integer('training_steps', 3000, "训练步数")

tf.flags.DEFINE_float('learning_rate_base', 0.001, "基础学习率")
tf.flags.DEFINE_string('dataset_name', 'mprh_part', "数据集名称")
FLAGS = tf.app.flags.FLAGS


def network_inference(images, num_class, arg_scope, func, is_training=True, dropout_keep_prob=0.8):
    with slim.arg_scope(arg_scope):
        return func(images, num_class, is_training=is_training, dropout_keep_prob=dropout_keep_prob)


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


def run(record_dir, save_model_path):
    if not tf.gfile.Exists(save_model_path):
        tf.gfile.MakeDirs(save_model_path)
    with tf.Graph().as_default():
        records_map_train, records_map_validation = input_data.get_records_map(record_dir)
        image_batch, label_batch = input_data.read_and_decode_mux(records_map_train, FLAGS.batch_size)
        val_image_batch, val_label_batch = input_data.read_and_decode_mux(records_map_validation, FLAGS.batch_size,
                                                                          is_training=False)
        images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
        labels = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='y-input')

        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=True, weight_decay=0.0004)
        logits, endpoints = network_inference(images, FLAGS.num_classes, arg_scope, mobilenet_v1.mobilenet_v1)
        exclusions = ['MobilenetV1/Logits']
        variables_to_restore = get_variables_to_restore(exclusions)
        cross_entropy = tools.cross_entropy_loss(logits=logits, labels=labels, weights_decay=1.0, name='Logits')
        # regularizer_loss = tools.regularizer_loss()
        total_loss = tools.total_loss()

        global_step = tf.Variable(0, trainable=False)

        learn_rate = FLAGS.learning_rate_base

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss=total_loss,
                                                                                 global_step=global_step)
        predict = endpoints['Predictions']
        accuracy = tools.accuracy(logits=predict, labels=labels)

        saver = tf.train.Saver(variables_to_restore)
        saver_last = tf.train.Saver(slim.get_model_variables(), max_to_keep=30)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_path)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('log/%s/train' % FLAGS.dataset_name, sess.graph)
            val_writer = tf.summary.FileWriter('log/%s/val' % FLAGS.dataset_name, sess.graph)

            i = 1
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                while not coord.should_stop() and i <= FLAGS.training_steps:
                    image_batch_value, label_batch_value = sess.run([image_batch, label_batch])
                    if i == 1 or i % 30 == 0:
                        summary_str = sess.run(summary_op, feed_dict={images: image_batch_value,
                                                                      labels: label_batch_value})
                        train_writer.add_summary(summary_str, i)
                    _, loss_value, accuracy_value = sess.run([train_op, cross_entropy, accuracy],
                                                             feed_dict={images: image_batch_value,
                                                                        labels: label_batch_value})
                    if i == 1 or i % 10 == 0:
                        print(i, 'loss:', loss_value, ', accuracy:', accuracy_value*100, '%')
                    if i == 1 or i % 50 == 0:
                        val_image_batch_value, val_label_batch_value = sess.run([val_image_batch, val_label_batch])
                        loss_value, accuracy_value = sess.run([cross_entropy, accuracy], feed_dict={
                            images: val_image_batch_value,
                            labels: val_label_batch_value
                        })
                        print(i, 'val loss:', loss_value, ', accuracy:', accuracy_value*100, '%')
                        summary_str = sess.run(summary_op, feed_dict={images: val_image_batch_value,
                                                                      labels: val_label_batch_value})
                        val_writer.add_summary(summary_str, i)
                    if i % 100 == 0:
                        saver_last.save(sess, os.path.join(save_model_path, 'model.ckpt'), global_step=i)
                    i += 1
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)


def main(_):
    for index in range(1):
        dataset_dir = 'dataset/%s/data_%d' % (FLAGS.dataset_name, index)
        model_save_path = 'models/%s/models_pro_k_fold/data_%d' + (FLAGS.dataset_name, index)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        run(dataset_dir, model_save_path)


if __name__ == '__main__':
    tf.app.run()
