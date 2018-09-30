from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
import sys

from six.moves import urllib


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """
  下载并解压数据集文件
  :param tarball_url: 数据集的url
  :param dataset_dir: 解压数据集输出的文件
  :return: 无返回直
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> 正在下载 %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('成功下载', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)
