import tensorflow as tf

"""
经典损失函数
"""
with tf.Session() as sess:
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f'tf.clip_by_value(v, 2.5, 4.5).eval()={tf.clip_by_value(v, 2.5, 4.5).eval()}')
    # [[ 2.5  2.5  3. ],[ 4.   4.5  4.5]] tf.clip_by_value函数可以将一个张量中的数值限定在一定范围内，避免log0无效或概率大于1
    v = tf.constant([1.0, 2.0, 3.0])
    print(f'tf.log(v).eval()={tf.log(v).eval()}')
    v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print(f'(v1 * v2).eval()={(v1 * v2).eval()}')  # *是元素直接相乘
    print(tf.matmul(v1, v2).eval())
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f'tf.reduce_mean(v).eval()={tf.reduce_mean(v).eval()}')
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits((y, y_)) # 带softmax结果的交叉熵
    # mse = tf.reduce_mean(tf.square(y_ - y))  回归问题对具体数值的预测，常用均方误差作为损失函数
