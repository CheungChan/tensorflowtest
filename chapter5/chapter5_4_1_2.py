import tensorflow as tf

"""
持久化代码的实现，定义了一样的声明变量，但是未初始化，使用保存的模型直接计算。
"""
# 使用和保存模型代码一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
saver = tf.train.Saver()
with tf.Session() as sess:
    # 加载已经存在的模型，并通过已经保存的模型中的变量的值来计算加法。
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run(result))
