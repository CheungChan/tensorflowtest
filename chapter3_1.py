import tensorflow as tf

"""
前向传播算法的使用
"""
# 声明w1 w2两个变量，这里还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵。
x = tf.constant([[0.7, 0.9]])
# 通过前向传播算法获得神经网络的输出。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
# 这里不能直接通过sess.run(y)计算y的取值，因为w1和w2还没有运行初始化过程。这里的两行分别初始化了w1和w2两个变量。
# sess.run(w1.initializer) # 初始化w1
# sess.run(w2.initializer) # 初始化w2
sess.run(tf.global_variables_initializer())
# 逐个变量调用initializer比较麻烦，可以用tf.global_variables_initializer初始化所有变量
# 原 tf.initialize_all_variables
print(sess.run(y))  # [[ 3.95757794]]
print(tf.global_variables())
# [<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(3, 1) dtype=float32_ref>]
print(tf.trainable_variables())
# [<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(3, 1) dtype=float32_ref>]
sess.close()
