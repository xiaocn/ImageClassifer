from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random

from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_name', 'floder', '数据集的名称, ["minst","cifar10", "flowers", "floder"]')
tf.app.flags.DEFINE_float('val_dp', 0.3, '验证集所占非测试集的比例')
tf.app.flags.DEFINE_float('test_dp', 0.1, '测试集所占全部数据比例')
tf.app.flags.DEFINE_integer('split_num', 1000, '文件的基本容量大小')
tf.app.flags.DEFINE_string('dataset_dir', None, '原始数据所存放的目录')
tf.app.flags.DEFINE_string('tfrecord_dir', None, 'tfrecord文件的保存路径')


def main(_):
  if not tf.gfile.Exists(FLAGS.tfrecord_dir):
    tf.gfile.MakeDirs(FLAGS.tfrecord_dir)
  photo_filenames, class_names = get_filenames_and_classes(FLAGS.dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  data_len = len(photo_filenames)
  test_num = int(data_len * FLAGS.test_dp)
  print(test_num)
  validation_num = int((data_len - test_num) * FLAGS.val_dp)
  num_share_dict ={'train': data_len-test_num-validation_num,
                   'validation': validation_num,
                   'test': test_num}

  if dataset_exists(FLAGS.tfrecord_dir, num_share_dict, FLAGS.split_num):
    print('tfrecord数据集已存在，不再重复创建！')
    return

  # 划分数据集为训练集，验证集，测试集
  random.seed(0)
  random.shuffle(photo_filenames)
  test_filenames = photo_filenames[:test_num]
  validation_filenames = photo_filenames[test_num:test_num+validation_num]
  training_filenames = photo_filenames[test_num + validation_num:]

  # 转换数据集
  if test_num < data_len:
    dataset_utils.write_tfrecord('train', training_filenames, class_names_to_ids, FLAGS.tfrecord_dir,
                     int(num_share_dict['train']/FLAGS.split_num)+1)
    if validation_num > 0:
      dataset_utils.write_tfrecord('validation', validation_filenames, class_names_to_ids, FLAGS.tfrecord_dir,
                       int(num_share_dict['validation']/FLAGS.split_num)+1)
  if test_num > 0:
    dataset_utils.write_tfrecord('test', test_filenames, class_names_to_ids, FLAGS.tfrecord_dir,
                     int(num_share_dict['test']/FLAGS.split_num)+1)

  # 创建标签文件
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_txtfile(labels_to_class_names, FLAGS.tfrecord_dir)

  #创建数据集数量说明文件
  dataset_utils.write_txtfile(num_share_dict, FLAGS.tfrecord_dir)
  print('\ntfrecord数据集转换完成!')


if __name__ == '__main__':
  tf.app.run()