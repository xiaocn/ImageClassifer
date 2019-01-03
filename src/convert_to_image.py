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
from matplotlib import pyplot as plt
from PIL import Image
import os

# 定义命令行参数
tf.flags.DEFINE_string('record_dir', None, "生成record文件的保存目录")
tf.flags.DEFINE_string('image_dir', None, "图像保存的路径")
tf.flags.DEFINE_string('split_name', 'train', "划分集名称，分为test，val，train三个")
tf.flags.DEFINE_integer('num', 100, "生成图像的数量")
FLAGS = tf.flags.FLAGS


def main(_):
    """
    用法说明： python3 convert_to_record.py \
                    --record_dir=<record_path> \
                    --image_dir=<image_path> \
                    [ --split_name=train \
                     --num=100
                    ]
        参数说名： record_dir —— 生成record文件的保存目录
                 image_dir —— 图像保存的路径
                 split_name —— 划分集合的名称，总共为三个：test，val，train，默认值为train
                 num —— 生成数据集的数量，默认值为100
            注： 中括号表示可选参数，等号后为默认值
    """
    record_file = os.path.join(FLAGS.record_dir, '%s_*.record' % FLAGS.split_name)
    files = tf.gfile.Glob(record_file)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    single_example = tf.parse_single_example(serialized_example, {
        'image': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
    })
    image = tf.decode_raw(single_example['image'], tf.uint8)
    height = tf.cast(single_example['height'], tf.int32)
    width = tf.cast(single_example['width'], tf.int32)
    channel = 3
    label = tf.cast(single_example['label'], tf.int32)
    image = tf.reshape(image, [height, width, channel])
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(FLAGS.num):
            plt.imshow(image.eval())
            plt.show()
            single, l = sess.run([image, label])
            img = Image.fromarray(single, 'RGB')
            img.save(os.path.join(FLAGS.image_dir, '%d-label-%d.jpg' % (i, l)))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
