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
import math
import random
import os
import sys
from PIL import Image

# 定义命令行参数
tf.flags.DEFINE_string('dataset_dir', None, "原始数据集的根目录")
tf.flags.DEFINE_string('record_dir', None, "生成record文件的保存目录")
tf.flags.DEFINE_float('test_dp', 0.1, "测试集占整个数据集的比例，默认值为0.1")
tf.flags.DEFINE_float('val_dp', 0.1, "验证集占训练集的比例，默认值为0.1")
tf.flags.DEFINE_integer('num_per_shard', 1000, "每个record文件所包含数据的数量，默认值为1000")
FLAGS = tf.flags.FLAGS


def get_dataset_and_labels(dataset_dir):
    """
    读取数据集中的标签及数据的文件路径
    :param dataset_dir: 数据集的根目录，该目录下分为n个子文件夹,每个文件夹代表一个类别，每个文件夹下有m张图像，表示该类别的数据
    :return: 返回标签列表及数据路径列表
    """
    photo_filenames = []
    class_names = []
    directories = os.listdir(dataset_dir)
    for directory in directories:
        path = os.path.join(dataset_dir, directory)
        if os.path.isdir(path):
            path_list = os.listdir(path)
            for filename in path_list:
                photo_filenames.append(os.path.join(path, filename))
            if len(path_list) > 0:
                class_names.append(directory)
    class_names = sorted(class_names)
    return photo_filenames, dict(zip(class_names, range(len(class_names))))


def dataset_exists(record_dir, num_shard_dict, num_per_shard):
    """
    判断record数据集是否存在
    :param record_dir: record数据集的根目录
    :param num_shard_dict: 各个划分集的数量
    :param num_per_shard: 每个record文件所含数据的数量
    :return: True-数据集文件已存在 False-不存在某些数据集文件
    """
    for split_name in num_shard_dict:
        num_shard = math.ceil(num_shard_dict[split_name] / num_per_shard)
        for shard_id in range(num_shard):
            record_file = os.path.join(record_dir, '%s_%05d-of-%05d.record' % (split_name, shard_id, num_shard))
            if not tf.gfile.Exists(record_file):
                return False
    return True


def main(_):
    """
    用法说明： python3 convert_to_record.py \
                    --dataset_dir=<dataset_path> \
                    --record_dir=<record_path> \
                    [--test_dp=0.1 \
                     --val_dp=0.1 \
                     --num_per_shard=1000
                    ]
        参数说名： dataset_dir —— 原始数据集的根目录，该目录下分为n个子文件夹,
                                每个文件夹代表一个类别，每个文件夹下有m张图像，表示该类别的数据集
                 record_dir —— 生成record文件的保存目录
                 test_dp —— 测试集占整个数据集的比例，默认值为0.1
                 val_dp —— 验证集占训练集的比例，默认值为0.1
                 num_per_shard —— 每个record文件所包含数据的数量，默认值为1000
            注： 中括号表示可选参数，等号后为默认值
    """
    filename_list, class_names_to_ids = get_dataset_and_labels(FLAGS.dataset_dir)
    data_len = len(filename_list)
    num_shard_dict = {
        'test': math.ceil(FLAGS.test_dp * data_len),
        'val': math.ceil((1 - FLAGS.test_dp) * data_len * FLAGS.val_dp),
        'train': math.ceil((1 - FLAGS.test_dp) * data_len * (1 - FLAGS.val_dp)),
    }
    if dataset_exists(FLAGS.record_dir, num_shard_dict, FLAGS.num_per_shard):
        print("数据集已存在，不再重复生成！")
    else:
        filename_dict = {}
        random.seed(0)
        random.shuffle(filename_list)
        start = 0
        for split_name in num_shard_dict:
            data_num = num_shard_dict[split_name]
            filename_dict[split_name] = filename_list[start:(data_num + start)]
            start = data_num
        if not tf.gfile.Exists(FLAGS.record_dir):
            tf.gfile.MakeDirs(FLAGS.record_dir)
        for split_name in filename_dict:
            filenames = filename_dict[split_name]
            num_shard = math.ceil(len(filenames) / FLAGS.num_per_shard)
            for shard_id in range(num_shard):
                output_filename = os.path.join(FLAGS.record_dir,
                                               '%s_%05d-of-%05d.record' % (split_name, shard_id, num_shard))
                with tf.python_io.TFRecordWriter(output_filename) as record_writer:
                    start_ndx = shard_id * FLAGS.num_per_shard
                    end_ndx = min((shard_id + 1) * FLAGS.num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> 正在装换%s数据集图像 %d/%d shard %d' %
                                         (split_name, i + 1, len(filenames), shard_id))
                        sys.stdout.flush()
                        image = Image.open(filenames[i])
                        image_data = image.tobytes()
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
                            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.size[1]])),
                            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.size[0]])),
                        }))
                        record_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
        print("数据集创建完毕！，详情请到目录%s下查看" % FLAGS.record_dir)


if __name__ == '__main__':
    tf.app.run()
