import tensorflow as tf

"""
使用字典来重命名变量
"""
# 这里声明的变量名称和已经保存的模型中的变量的名称不同。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = v1 + v2
# 如果直接使用tf.train.Saver()来加载模型会报变量找不到的错误.
# tensorflow.python.framework.errors_impl.NotFoundError: Key other-v1 not found in checkpoint
# saver = tf.train.Saver()
saver = tf.train.Saver({"v1": v1, "v2": v2})
with tf.Session() as sess:
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run(result))
