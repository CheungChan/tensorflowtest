import tensorflow as tf

"""
使用saver = tf.train.Saver([v1])来加载部分变量。
"""
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
saver = tf.train.Saver([v1])
with tf.Session() as sess:
    # 加载已经存在的模型，并通过已经保存的模型中的变量的值来计算加法。
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run(result))
    # 报错tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value v2
