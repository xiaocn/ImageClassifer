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
import os
import math
import sys
import dataset_utils
import random

# 自定义参数
tf.flags.DEFINE_integer('num_share', 5, "record文件数量")
tf.flags.DEFINE_integer('data_k', 6, "数据段数量")
tf.flags.DEFINE_string('dataset_name', 'mprh_part', "数据集名称")
FLAGS = tf.app.flags.FLAGS


class ImageReader(object):

    def __init__(self):
        self.__decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self.__decode_jpeg = tf.image.decode_jpeg(self.__decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self.__decode_jpeg, feed_dict={self.__decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'images_%s_%05d-of-%05d.record' % (split_name, shard_id, FLAGS.num_shard)
    return os.path.join(dataset_dir, output_filename)


def _get_filenames_map_and_classes(dataset_dir):
    warehouse_root = dataset_dir
    directories = []
    class_names = []
    for filename in os.listdir(warehouse_root):
        path = os.path.join(warehouse_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    image_filenames_map = {}
    for index, directory in enumerate(directories):
        class_name = class_names[index]
        image_filenames_map[class_name] = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            image_filenames_map[class_name].append(path)
    return image_filenames_map, sorted(class_names)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(FLAGS.num_shard)))

    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session() as sess:

            for shard_id in range(FLAGS.num_shard):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Convert image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        example = dataset_utils.image_to_example(image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(_):
    dataset_dir = FLAGS.dataset_name
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if tf.gfile.Exists(os.path.join(dataset_dir, dataset_utils.LABELS_FILENAME)):
        print('Dataset files already exist. Exiting without re-creating them')
        return

    # image_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    image_filenames_map, class_names = _get_filenames_map_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    for class_name in class_names:
        image_filenames = image_filenames_map[class_name]

        # Divide into train and test:
        num_validation = int(math.ceil(len(image_filenames) / FLAGS.data_k))
        random.seed(0)
        random.shuffle(image_filenames)

        validation_start = 0
        validation_end = num_validation
        for k_fold_index in range(FLAGS.data_k):
            validation_filenames = image_filenames[validation_start:validation_end]
            train_filenames = image_filenames[:validation_start] + image_filenames[validation_end:]

            validation_start += num_validation
            validation_end += num_validation

            dataset_dir_k_fold = os.path.join(dataset_dir, 'data_%d' % k_fold_index)
            if not tf.gfile.Exists(dataset_dir_k_fold):
                tf.gfile.MakeDirs(dataset_dir_k_fold)

            dataset_dir_single_class = os.path.join(dataset_dir_k_fold, class_name + '_record')
            if not tf.gfile.Exists(dataset_dir_single_class):
                tf.gfile.MakeDirs(dataset_dir_single_class)

            # First, convert the training and validation sets.
            _convert_dataset('train', train_filenames, class_names_to_ids, dataset_dir_single_class)
            _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir_single_class)

    # Finally, write the labels file:
    lables_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(lables_to_class_names, dataset_dir)

    print('\nFinished converting the dataset')


if __name__ == '__main__':
    tf.app.run()
