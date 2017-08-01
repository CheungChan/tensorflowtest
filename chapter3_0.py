import tensorflow as tf

"""
TensorFlow安装后的测试样例
"""
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
sess = tf.Session()
print(sess.run(result))
sess.close()
"""
TensorFlow的计算模型是计算图，使用tf.get_default_graph()获得默认的计算图，使用tf.Tensor.graph获得某个Tensor对应的计算图。
"""
# 通过a.graph可以查看张量所属的计算图。因为没有特意指定，所以这个计算图应该属于当前的默认计算图。所以下面这个操作输出为True。
print(a.graph is tf.get_default_graph())
"""
不同计算图中的张量不能共享。
"""
g1 = tf.Graph()
with g1.as_default():
    # 在计算图个g1中定义变量"v",并设置初始值为0
    v1 = tf.get_variable("v", initializer=tf.zeros_initializer()([1]))  # 原tf.zerors_initializer(shape=[1])改
g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量"v",并设置初始值为1.
    v2 = tf.get_variable("v", initializer=tf.ones_initializer()([1]))  # 原tf.ones_initializer(shape=[1])改
# 在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()  # 原tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，变量“v“的取值应该为0，所以下面这行会输出[0.].
        print(sess.run(tf.get_variable("v")))  # v1
# 在计算图g2中读取变量“v”的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()  # 原tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g2中，变量"v"的取值应该为1，所以下面这行会输出[1.]。
        print(sess.run(tf.get_variable("v")))  # v2
g = tf.Graph()
"""
使用计算图可以指定计算运行的设备
"""
with g.device('/gpu:0'):
    result = a + b
    with tf.Session() as sess:
        print(sess.run(result))
"""
TensorFlow的数据模型张量的概念
"""
result = tf.add(a, b, name="add")
print(result)  # 输出 Tensor("add_2:0", shape=(2,), dtype=float32)，表明TensorFlow计算的结果不是一个具体的数字，而是一个张量结构。
# 一个张量主要保存三个属性：名字name，维度shape，类型dtype。名字：node:src_output，node为节点名称，src_output表示当前张量来自节点的
# 第几个输出。
"""
TensorFlow的运行模型会话
"""
sess = tf.Session()
with sess.as_default():
    print(result.eval())
    # 当默认的会话指定时，可以用tf.Tensor.eval()来计算张量的取值。
    # sess.run(result) = result.eval(session=sess)
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
# 在交互式环境下，直接构建默认会话，操作更为方便。
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# sess1 = tf.InteractiveSession(config=config)
# sess2 = tf.Session(config=config)
# # 无论哪种session都可以通过ConfigProto配置会话。
