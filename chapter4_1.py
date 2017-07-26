import tensorflow as tf

with tf.Session() as sess:
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(tf.clip_by_value(v, 2.5, 4.5).eval())
    v = tf.constant([1.0, 2.0, 3.0])
    print(tf.log(v).eval())
    v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print((v1 * v2).eval())
    print(tf.matmul(v1, v2).eval())
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(tf.reduce_mean(v).eval())
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits((y, y_))
