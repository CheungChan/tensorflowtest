import tensorflow as tf

"""
通过变量重命名直接读取变量的滑动平均值
"""
v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run(v))

