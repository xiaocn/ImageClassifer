from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def int64_feature(values):
  """
  64位整数列表编码器
  :param values: 64位的整数序列
  :return: 返回t64位整数列表的编码序列
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """
  字节编码器
  :param values: 字节序列
  :return: 返回字节列表的编码序列
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """
  浮点数编码器
  :param values: 浮点数序列
  :return: 返回浮点数的编码序列
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  """
  图像编码器
  :param image_data: 输入的图像数组
  :param image_format: 图像的编码格式
  :param height: 图像的高
  :param width: 图像的宽
  :param class_id: 该图像对应的标签
  :return: 返回图像的编码序列
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def keys_to_features():
  keys_feature = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }
  return keys_feature


def items_to_handlers():
  items_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }
  return items_handlers


def items_to_desc(split_name_dict):
  items_desc_dict = {
    'image':split_name_dict['image'],
    'label':split_name_dict['label'],
  }
  return items_desc_dict