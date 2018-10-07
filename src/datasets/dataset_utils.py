from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import feature_utils


def get_file_name(tfrecord_dir, split_name, shard_id=-1, shard_num=-1):
  data_name = tfrecord_dir.split('/')[-1]
  label_file = os.path.join(tfrecord_dir, "%s_labels.txt" % data_name)
  num_file = os.path.join(tfrecord_dir, "%s_num.txt" % data_name)
  tfrecord_file = os.path.join(tfrecord_dir, "%s_%s_*.tfrecord" % (data_name, split_name))
  if shard_num > 0:
    tfrecord_file = os.path.join(tfrecord_dir, "%s_%s_%05d-of-%05d.tfrecord" %
                                 (data_name, split_name, shard_id, shard_num))
  return tfrecord_file,label_file,num_file


def write_tfrecord(split_name, filenames, class_names_to_ids, tfrecord_dir, num_shares):
  """
  转换tfrecord数据集
  :param split_name: 划分名称，分别为test,validation,train三类
  :param filenames: 数据集文件名列表
  :param class_names_to_ids: 标签对应的下标
  :param tfrecord_dir: tfrecord文件的保存目录
  :param num_shares: tfrecord文件的数量
  :return: 无返回
  """
  assert split_name in ['train', 'validation', 'test']
  num_per_shard = int(math.ceil(len(filenames) / float(num_shares)))
  with tf.Graph().as_default():
    decode_jpeg_data = tf.placeholder(dtype=tf.string)
    decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
    with tf.Session('') as sess:
      for shard_id in range(num_shares):
        data_name = tfrecord_dir.split('/')[-1]
        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (data_name, split_name, shard_id, num_shares)
        output_filename = os.path.join(tfrecord_dir, output_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
            height, width = image.shape[0], image.shape[1]
            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]
            example = feature_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()


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

