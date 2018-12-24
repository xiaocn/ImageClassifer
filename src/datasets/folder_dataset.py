#!/bin/usr/python3
# -*- coding: UTF-8 -*-

"""
模块说明： 主要定义以文件夹作为分类名的图像分类数据集
作者： xiao
时间： 2018.12.23
"""

# 导入future模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入自定义模块
import os
import random
import tensorflow as tf
from PIL import Image


dataset_desc_dict = {
    'image': "任意大小的彩色图像",
    'label': "标签下标值为从0到大于0的任意整数范围",
}


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
    random.seed(0)
    random.shuffle(photo_filenames)
    return photo_filenames, dict(zip(class_names, range(len(class_names))))


def get_image_and_label(filenames, class_names_to_ids, index):
    """
    返回单张图像及标签
    :param filenames: 数据文件列表
    :param class_names_to_ids: 标签列表
    :param index: 数据的下标
    :return: 返回第index张图像及标签
    """
    image_data = tf.gfile.FastGFile(filenames[index], 'rb').read()
    image = Image.open(filenames[index])
    class_name = os.path.basename(os.path.dirname(filenames[index]))
    class_id = class_names_to_ids[class_name]
    return image_data, image.size[0], image.size[1], class_id
