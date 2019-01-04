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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops


def apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                   for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.25)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, fast_mode=True, central_fraction=0.95, scope=None):
    with tf.name_scope(scope, 'distort_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(image, lambda x, method:
                                                     tf.image.resize_images(x, [height,width], method),
                                                     num_cases=num_resize_cases)
        tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        rot_k = random_ops.random_uniform([], 0, 3, dtype=tf.int32)
        distorted_image = tf.image.rot90(distorted_image, rot_k)

        tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))

        return distorted_image


def preprocess_for_eval(image, height, width, central_fraction=0.95, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
        return image


def preprocess_for_store(image, height, width, central_fraction=0.95):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.multiply(image, 0.00392157)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    return image


def preprocess(image, height, width, scope=None, is_training=False):
    if is_training:
        return preprocess_for_train(image, height, width)
    else:
        return preprocess_for_eval(image, height, width)