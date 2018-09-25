from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import convert_floder_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_name', 'floder', '数据集的名称, ["minst","cifar10", "flowers", "floder"]')
tf.app.flags.DEFINE_float('val_dp', 0.3, '验证集所占非测试集的比例')
tf.app.flags.DEFINE_float('test_dp', 0.1, '测试集所占全部数据比例')
tf.app.flags.DEFINE_integer('split_num', 1000, '文件的基本容量大小')
tf.app.flags.DEFINE_string('dataset_dir', None, '原始数据所存放的目录')
tf.app.flags.DEFINE_string('tfrecord_dir', None, 'tfrecord文件的保存路径')

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('请设置原始数据目录')

  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'floder':
    if not FLAGS.tfrecord_dir:
      raise ValueError('请设置tfrecord文件的保存路径')
    convert_floder_data.run(FLAGS.dataset_dir,
                            FLAGS.tfrecord_dir,
                            FLAGS.val_dp,
                            FLAGS.test_dp,
                            FLAGS.split_num)
  else:
    raise ValueError(
        'dataset_name [%s] 不存在.' % FLAGS.dataset_name)

if __name__ == '__main__':
  data = {}
  tf.app.run()