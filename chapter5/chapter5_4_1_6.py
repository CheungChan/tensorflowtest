import tensorflow as tf

"""
持久化滑动平均的变量
"""
v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有声明滑动平均模型时只有一个变量v，所有输出v:0
for variables in tf.global_variables():
    print(variables.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型之后，TensorFlow会自动生成一个影子变量v/ExponentialMoving Average。于是输出v:0和v/ExponentialMovingAverage:0
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时，Tensorflow会将v:0和v/ExponentialMovingAverage:0两个变量都存起来。
    saver.save(sess, r"D:/tmp/model/model.ckpt")
    print(sess.run([v, ema.average(v)]))
