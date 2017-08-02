import tensorflow as tf

"""
通过TensorFlow训练神经网络模型
"""
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# tf.assign(w1, w2, validate_shape=False)
# x = tf.constant([[0.7, 0.9]])
# 定义placeholder作为存放输入数据的地方。这里维度也不一定要定义。但如果维度是确定的，那么给出维度可以降低出错的概率。
x = tf.placeholder(tf.float32, shape=(None, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# print(sess.run(y))
# You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
# 会跟上一节输出一样的结果 [[ 3.95757794]]
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
# [[ 3.95757794]
#  [ 1.15376544]
#  [ 3.16749239]]
sess.close()
