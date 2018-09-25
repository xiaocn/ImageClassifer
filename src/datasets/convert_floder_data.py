from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0


class ImageReader(object):
  """
  图像读取的对象
  """
  def __init__(self):
    # 初始化类，定义解码器及图像数据类型
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """
  读取数据集中的标签及数据的文件路径
  :param dataset_dir: 数据集的根目录，该目录下分为n个子文件夹,每个文件夹代表一个类别，每个文件夹下有m张图像，表示该类别的数据
  :return: 返回标签列表及数据路径列表
  """
  directories = []
  class_names = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(tfrecord_dir, split_name, shard_id, num_shares):
  """
  获取tfrecord文件的文件路径（包含文件名）
  :param tfrecord_dir: tfrecord文件的根目录
  :param split_name: 划分数据集的名称，只有train,validation,test三类
  :param shard_id: 第几个tfrecord文件
  :param num_shares: tfrecord文件的总数
  :return: 返回tfrecord文件的路径（包含文件名）
  """
  data_name = tfrecord_dir.split('/')[-1]
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (data_name, split_name, shard_id, num_shares)
  return os.path.join(tfrecord_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, tfrecord_dir, num_shares):
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
    image_reader = ImageReader()
    with tf.Session('') as sess:
      for shard_id in range(num_shares):
        output_filename = _get_dataset_filename(tfrecord_dir, split_name, shard_id, num_shares)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(tfrecord_dir, num_shares_dict, split_num):
  """
  判断tfrecord数据集是否存在
  :param tfrecord_dir: tfrecord数据集的根目录
  :param num_shares_dict: tfrecord数据集各个划分集的文件总数量
  :param split_num: 每个tfrecord文件的容量大小
  :return: True-文件已存在 False-不存在某些数据集文件
  """
  for split_name in ['train', 'validation', 'test']:
    num_shard = int(num_shares_dict[split_name] / split_num) + 1
    for shard_id in range(num_shard):
      output_filename = _get_dataset_filename(tfrecord_dir, split_name, shard_id, num_shard)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, tfrecord_dir, validation_dp=0.3, test_dp=0.1, split_num=1000):
  """
  转换数据集，将文件夹（文件夹作为类别）中的数据集转换为tfrecord数据集
  :param dataset_dir: 原始数据，该目录下有n个文件夹，表示n个类别，其中每个文件夹下有m张图像，表示该类别的m个数据
  :param tfrecord_dir: tfrecord格式文件的存放路径
  :param validation_dp: 验证集占非测试集的比例
  :param test_dp: 测试集占总数据的比例
  :return: 无返回
  """

  if not tf.gfile.Exists(tfrecord_dir):
    tf.gfile.MakeDirs(tfrecord_dir)

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  data_len = len(photo_filenames)
  test_num = int(data_len * test_dp)
  print(test_num)
  validation_num = int((data_len - test_num) * validation_dp)

  num_share_dict ={'train': data_len-test_num-validation_num,
                   'validation': validation_num,
                   'test': test_num}

  if _dataset_exists(tfrecord_dir, num_share_dict, split_num):
    print('tfrecord数据集已存在，不再重复创建！')
    return

  # 划分数据集为训练集，验证集，测试集
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  test_filenames = photo_filenames[:test_num]
  validation_filenames = photo_filenames[test_num:test_num+validation_num]
  training_filenames = photo_filenames[test_num + validation_num:]

  # 转换数据集
  if test_num < data_len:
    _convert_dataset('train', training_filenames, class_names_to_ids, tfrecord_dir,
                     int(num_share_dict['train']/split_num)+1)
    if validation_num > 0:
      _convert_dataset('validation', validation_filenames, class_names_to_ids, tfrecord_dir,
                       int(num_share_dict['validation']/split_num)+1)
  if test_num > 0:
    _convert_dataset('test', test_filenames, class_names_to_ids, tfrecord_dir,
                     int(num_share_dict['test']/split_num)+1)

  # 创建标签文件
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

  #创建数据集数量说明文件
  dataset_utils.write_datanum_file(num_share_dict, tfrecord_dir)
  print('\ntfrecord数据集转换完成!')