#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要实现模型的读写，配置等
作者： xiao
时间： 2018.12.23
"""

# 导入future模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入系统模块
import tensorflow as tf


def load_single_pb(model_file):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
    for node in graph_def.node:
        if 'rfcn_' in model_file and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'faster_rcnn_' in model_file and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def init_graph_ops(graph, name_list):
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in name_list:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
    return tensor_dict
