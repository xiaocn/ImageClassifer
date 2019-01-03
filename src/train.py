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
import numpy as np

# 定义命令行参数
tf.flags.DEFINE_string('record_dir', None, "生成record文件的保存目录")
tf.flags.DEFINE_string('image_dir', None, "图像保存的路径")
tf.flags.DEFINE_string('split_name', 'train', "划分集名称，分为test，val，train三个")
tf.flags.DEFINE_integer('train_step', 100, "生成图像的数量")
tf.flags.DEFINE_float('learning_rate', 0.001, "学习率")
FLAGS = tf.flags.FLAGS


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image, (height, width), method=np.random.randint(4))

    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(3))
    return distorted_image


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(image_batch, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([784, 500], regularizer)
        biases = tf.get_variable('biases', [500], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(image_batch, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([500, 5], regularizer)
        biases = tf.get_variable('biases', [5], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2


def main(_):
    """
    用法说明： python3 convert_to_record.py \
                    --record_dir=<record_path> \
                    --image_dir=<image_path> \
                    [ --split_name=train \
                     --train_step=100
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
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    image_size = 300
    distort_image = preprocess_for_train(image, image_size, image_size, boxes)
    min_after_dequeue = 10000
    batch_size = 100
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([distort_image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)
    logit = inference(image_batch)
    loss = calc_loss(logit, label_batch)
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(FLAGS.train_step):
            result = preprocess_for_train(image, 300, 300, boxes)
            plt.imshow(result.eval())
            plt.show()
            single, l = sess.run([result, label])
            img = Image.fromarray(single, 'RGB')
            img.save(os.path.join(FLAGS.image_dir, '%d-label-%d.jpg' % (i, l)))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
