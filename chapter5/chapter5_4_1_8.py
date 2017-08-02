import tensorflow as tf

"""
使用variables_to_restore函数来生成变量重命名字典
"""
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run(v))
