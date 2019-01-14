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
import os
import tensorflow.contrib.slim as slim

# 导入自定义模块
import image_preprocessing
import mobilenet_v1


def network_fn(images, num_classes, arg_scope, func, is_training=True):
    with slim.arg_scope(arg_scope):
        return func(images, num_classes, is_training=is_training)


image_data = tf.placeholder(dtype=tf.string, name='input')

image_decoded = tf.image.decode_jpeg(image_data, channels=3)
image_decoded = image_preprocessing.preprocess_for_store(image_decoded, 224, 224)

arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)
logits, endpoints = network_fn(image_decoded, 2, arg_scope, mobilenet_v1.mobilenet_v1, is_training=False)

predict = endpoints['Predictions']
predict = tf.reshape(predict, [1, 2], name='output')

saver = tf.train.Saver(slim.get_model_variables())
saver_store = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    model_save_path = './small/models_pro_k_fold/k_fold_0'
    model_saved_name = 'model.ckpt-3800'
    log_path = 'small/log_store/'

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver.restore(sess, os.path.join(model_save_path, model_saved_name))

    writer = tf.summary.FileWriter(log_path, sess.graph)

    image_filename = './small_test/clear/IMG_1228.JPG_C3_clear.jpg'
    image_data_value = tf.gfile.FastGFile(image_filename, 'rb').read()
    predict_value = sess.run(predict, feed_dict={image_data: image_data_value})
    print(predict_value)

    model_name = 'model_store.ckpt'
    tf.train.write_graph(sess.graph_def, model_save_path, 'model_graph.pb', as_text=True)
    saver_store.save(sess, os.path.join(model_save_path, model_name))
