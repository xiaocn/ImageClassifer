import tensorflow as tf
from tensorflow.python.framework import graph_util


if __name__ == '__main__':
    output_node_names = "output"
    input_checkpoint = 'small/models_pro_k_fold/k_fold_0/model_store.ckpt'
    output_graph = 'small/models_pro_k_fold/k_fold_0/freeze_graph_mprh.pb'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) 
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, input_graph_def=sess.graph_def, output_node_names=output_node_names.split(","))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
