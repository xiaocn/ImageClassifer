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
from collections import namedtuple
import functools


Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]


def mobilenet_v1_base(inputs, final_endpoint='Conv2d_13_pointwise', min_depth=8, depth_multiplier=1.0, conv_defs=None,
                      output_stride=None, scope=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            current_stride = 1
            rate = 1
            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not  None and current_stride == output_stride:
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    net = slim.separable_conv2d(net, None, conv_def.kernel, depth_multiplier=1, stride=layer_stride,
                                                rate=layer_rate, normalizer_fn=slim.batch_norm, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1], stride=1, normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknow convolution type %s for layer %d' % (conv_def.ltype, i))
    raise ValueError('Unknow final endpoint %s' % final_endpoint)


def mobilenet_v1(inputs, num_classes=1000, dropout_keep_prob=0.999, is_training=True, min_depth=8, depth_multiplier=1.0,
                 conv_defs=None, prediction_fn=tf.contrib.layers.softmax, spatial_squeeze=True, reuse=None,
                 scope='MobilenetV1'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))
    with tf.variable_scope(scope, 'MobilenetV1', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v1_base(inputs,scope=scope, min_depth=min_depth,
                                                depth_multiplier=depth_multiplier, conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                end_points['AvgPool_1a'] = net

                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                     scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


mobilenet_v1.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


mobilenet_v1_075 = wrapped_partial(mobilenet_v1, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet_v1, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet_v1, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
    return kernel_size_out


def mobilenet_v1_arg_scope(is_training=True, weight_decay=0.00004, stddev=0.09, regularize_depthwise=False):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.99,
        'epsilon': 0.001,
    }
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
                    return sc

