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
from tensorflow.examples.tutorials.mnist import input_data
import time

#导入自定义库
import net


def main(_):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    y = net.inference(x, None)
    mnist = input_data.read_data_sets('/ai/workrooms/datasets/org-img/bak', one_hot=True)
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_averages = tf.train.ExponentialMovingAverage(0.99)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    pre_step = ''
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('/ai/workrooms/datasets/test/model/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                if global_step == pre_step:
                    break
                else:
                    pre_step = global_step
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print('after %s train step, validation accuray is %g' % (global_step, accuracy_score))
                if accuracy_score > 0.5:
                    print('----------------------------------yes-----------------------------')
            else:
                print('No checkpoint file found')
                break
            time.sleep(10)


if __name__ == '__main__':
    tf.app.run()