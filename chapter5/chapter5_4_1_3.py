import tensorflow as tf

"""
不重新定义变量，直接加载持久化的图，来加载所有变量
"""
saver = tf.train.import_meta_graph(r"D:/tmp/model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
