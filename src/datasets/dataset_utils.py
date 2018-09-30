from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import feature_utils


def get_file_name(tfrecord_dir, split_name, shard_id=-1, shard_num=-1,
                  label_name="labels.txt", num_name="num.txt"):
  data_name = tfrecord_dir.split('/')[-1]
  label_file = os.path.join(tfrecord_dir,data_name+label_name)
  num_file = os.path.join(tfrecord_dir,data_name+num_name)
  tfrecord_file = os.path.join(tfrecord_dir,"%s_%s_*.tfrecord" % (data_name, split_name))
  if shard_num > 0:
    tfrecord_file = os.path.join(tfrecord_dir,"%s_%s_%05d-of-%05d.tfrecord" %
                                 (data_name, split_name, shard_id, shard_num))
  return tfrecord_file,label_file,num_file


def read_tfrecord(tfrecord_dir, split_name, reader=tf.TFRecordReader):
  tfrecord_file, label_file, num_file = get_file_name(tfrecord_dir,split_name)
  if not (tf.gfile.Exists(num_file) and tf.gfile.Exists(label_file)):
    raise ValueError("%s没有缺失标签文件或样本数量说明文件" % tfrecord_dir)
  label_name_dict = read_txtfile(label_file)
  split_name_dict = read_txtfile(num_file)
  item_desc_dict = feature_utils.items_to_desc(split_name_dict)
  keys_to_features = feature_utils.keys_to_features()
  items_to_handlers = feature_utils.items_to_handlers()
  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
  return slim.dataset.Dataset(
      data_sources=tfrecord_file,
      reader=reader,
      decoder=decoder,
      num_samples=split_name_dict[split_name],
      items_to_descriptions=item_desc_dict,
      num_classes=len(label_name_dict),
      labels_to_names=label_name_dict)


def write_txtfile(data, filename):
  """
  将标签数据或其他数据写入文本文件
  :param data: 要写入文件的数据
  :param filename: 文件输出路径（含文件名）
  :return: 无返回值
  """
  with tf.gfile.Open(filename, 'w') as f:
    for i in data:
      item = data[i]
      f.write('%s:%d\n' % (item, i))


def read_txtfile(filename):
  """
  读取文本文件
  :param filename: 文件名（含路径）
  :return: 返回以分割符分开的数据字典
  """
  with tf.gfile.Open(filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)
  data = {}
  for line in lines:
    index = line.index(':')
    data[int(line[:index+1])] = line[index:]
  return data

