import tensorflow as tf
from numpy.random import RandomState

"""
完整的神经网络样例程序
"""
# numpy是一个科学计算的工具包。这里通过numpy工具包生成模拟数据集。
# 定义训练数据batch的大小。
batch_size = 8
# 定义神经网络中的参数。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 在shape的一个维度上使用None可以方便使用不大的batch大小。在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。
# 当测试集比较小时，这样比较方便测试，但数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义神经网络的前向传播过程。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数和反向传播的算法。
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 通过随机数生成一个模拟数据集。
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# X = array([[  4.17022005e-01,   7.20324493e-01],[  1.14374817e-04,   3.02332573e-01],.....)
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），而其他为负样本（比如零件不合格）。和TensorFlow
# 游乐场中的表示法不大一样的地方是，在这里使用0来表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用0和1表示法。
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
# 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)
    print(f"w1 = {sess.run(w1)}")
    print(f"w2 = {sess.run(w2)}")
    """在训练之前神经网络参数的值
    w1 = [[-0.81131822  1.48459876  0.06532937]
     [-2.4427042   0.0992484   0.59122431]]
    w2 = [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]

    """
    # 设定训练的轮数
    STEP = 5000
    for i in range(STEP):
        # 每次选定batch_size个样本进行训练。
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # print(f"start = {start}, end = {end}")
        # 通过选取的样本训练神经网络并更新参数。
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print(f"After {i} traning step(s), cross entropy on all data is {total_cross_entropy}")
            """
            After 0 traning step(s), cross entropy on all data is 0.06749248504638672
            After 1000 traning step(s), cross entropy on all data is 0.01633850485086441
            After 2000 traning step(s), cross entropy on all data is 0.009075473994016647
            After 3000 traning step(s), cross entropy on all data is 0.007144360803067684
            After 4000 traning step(s), cross entropy on all data is 0.005784708075225353
            通过这个结果可以发现随着训练的进行，交叉熵是逐渐变小的。交叉熵越小说明预测的结果和真实的结果差距越小。
            """
    print(f"w1 = {sess.run(w1)}")
    print(f"w2 = {sess.run(w2)}")
    """
    在训练后神经网络参数中的值
    w1 = [[-1.96182752  2.58235407  1.68203771]
     [-3.46817183  1.06982315  2.11788988]]
    w2 = [[-1.82471502]
     [ 2.68546653]
     [ 1.41819501]]
     可以发现这两个参数的取值已经发生了变化，这个变化就是训练的结果。
     它使得这个神经网络能更好的拟合提供的训练数据。
     """
