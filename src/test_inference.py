import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile
import time as t


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def 
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def show_node(graph):
    for op in graph.get_operations():
        print(op.name)


if __name__ == '__main__':

    graph = load_graph('/ai/workrooms/CNNG/mprh_classifier/total/models_pro_k_fold/k_fold_0/freeze_graph_mprh.pb')
    input_tensor = graph.get_tensor_by_name('prefix/input:0')
    output_tensor = graph.get_tensor_by_name('prefix/output:0')
    basepath = './total_test'
    with tf.Session(graph=graph) as sess:
        total = 0
        correct = 0
        muddy_path = os.path.join(basepath,'muddy')
        clear_path = os.path.join(basepath,'clear')
        muddy_images = os.listdir(muddy_path)
        clear_images = os.listdir(clear_path)
        start = t.time()
        for image_filename in muddy_images:
            image_data = gfile.FastGFile(os.path.join(muddy_path, image_filename), 'rb').read()
            predict = sess.run(output_tensor, feed_dict={input_tensor: image_data})
            predict_label = np.argmax(predict[0])
            if predict_label == 1:
                correct += 1
            total += 1

        for image_filename in clear_images:
            image_data = gfile.FastGFile(os.path.join(clear_path, image_filename), 'rb').read()
            predict = sess.run(output_tensor, feed_dict={input_tensor: image_data})
            predict_label = np.argmax(predict[0])
            if predict_label == 0:
                correct += 1
            total += 1
        end = t.time()
        print(correct)
        print(total)
        print(correct / total)
        print('mean time is %.2f ms' % ((end-start) * 1000 / total))

