from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_floder_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_name', 'floder', '数据集的名称, ["minst","cifar10", "flowers", "floder"]')
tf.app.flags.DEFINE_float('val_dp', 0.3, '验证集所占非测试集的比例')
tf.app.flags.DEFINE_float('test_dp', 0.1, '测试集所占全部数据比例')
tf.app.flags.DEFINE_integer('split_num', 1000, '文件的基本容量大小')
tf.app.flags.DEFINE_string('dataset_dir', None, '原始数据所存放的目录')
tf.app.flags.DEFINE_string('tfrecord_dir', None, 'tfrecord文件的保存路径')


def main(_):
  assert split_name in ['train', 'validation', 'test']
  num_per_shard = int(math.ceil(len(filenames) / float(num_shares)))
  with tf.Graph().as_default():
    decode_jpeg_data = tf.placeholder(dtype=tf.string)
    decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
    with tf.Session('') as sess:
      for shard_id in range(num_shares):
        data_name = tfrecord_dir.split('/')[-1]
        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (data_name, split_name, shard_id, num_shares)
        output_filename = os.path.join(tfrecord_dir, output_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
            height, width = image.shape[0], image.shape[1]

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = feature_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


if __name__ == '__main__':
  tf.app.run()