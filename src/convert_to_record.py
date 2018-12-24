#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块功能： 实现将以文件夹为分类标签的数据集转换为record数据集
作者： xiao
时间： 2018.12.21
"""

# 导入future模块
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 导入系统模块
import tensorflow as tf

# 导入自定义模块
import factory as factory
from datasets import dataset_utils as utils

# 定义命令行参数
tf.flags.DEFINE_string('dataset_dir', None, "原始数据集的根目录")
tf.flags.DEFINE_string('record_dir', None, "生成record文件的保存目录")
tf.flags.DEFINE_string('dataset_name', None, "数据集文件名，目前只有flower")
tf.flags.DEFINE_float('test_dp', 0.1, "测试集占整个数据集的比例，默认值为0.1")
tf.flags.DEFINE_float('val_dp', 0.1, "验证集占训练集的比例，默认值为0.1")
tf.flags.DEFINE_integer('num_per_shard', 1000, "每个record文件所包含数据的数量，默认值为1000")
FLAGS = tf.flags.FLAGS


def main(_):
    """
    用法说明： python3 convert_to_record.py \
                    --dataset_dir=<dataset_path> \
                    --record_dir=<record_path> \
                    --dataset_name=<dataset_name>
                    [--test_dp=0.1 \
                     --val_dp=0.1 \
                     --num_per_shard=1000
                    ]
        参数说名： dataset_dir —— 原始数据集的根目录，该目录下分为n个子文件夹,
                                每个文件夹代表一个类别，每个文件夹下有m张图像，表示该类别的数据集
                 record_dir —— 生成record文件的保存目录
                 dataset_name —— 数据集名称
                 test_dp —— 测试集占整个数据集的比例，默认值为0.1
                 val_dp —— 验证集占训练集的比例，默认值为0.1
                 num_per_shard —— 每个record文件所包含数据的数量，默认值为1000
            注： 中括号表示可选参数，等号后为默认值
    """
    dataset = factory.get_dataset(FLAGS.dataset_name)
    filename_list, class_names_to_ids = dataset.get_dataset_and_labels(FLAGS.dataset_dir)
    filename_dict = utils.get_data_partition(filename_list, test_dp=FLAGS.test_dp, val_dp=FLAGS.val_dp)
    if utils.dataset_exists(FLAGS.record_dir, filename_dict, FLAGS.num_per_shard, class_names_to_ids):
        print("数据集已存在，不再重复生成！")
    else:
        utils.create_dataset(dataset, FLAGS.record_dir, filename_dict, class_names_to_ids,
                             num_per_shard=FLAGS.num_per_shard)


if __name__ == '__main__':
    tf.app.run()
