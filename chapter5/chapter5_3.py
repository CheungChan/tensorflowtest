import tensorflow as tf

"""
使用tf.get_variable和tf.Variable创建变量是一致的，但是get_variable的名称是必填项，Variable的name名称不是必填项。
tf.get_variable如果变量名称已经存在会报错，不允许重用，如果想获取变量值可使用tf.variable_scope创建一个上下文管理器并设置reuse
=True,再使用tf.get_variable获取变量。同时设置了reuse=True的variable_scope只能获取已经存在的变量否则报错，不能创建。
"""
# 下面这两个定义是等价的
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
print(v)
v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
print(v)

# 再名字为foo的命名空间内创建名字为v的变量。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所以下面的代码会报错。
# Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置为True，这样tf.get_variable函数将直接获取已经声明的变量。
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)  # True
# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。因为在命名空间Bar中还没有创建变量v，所以下面代码会报错。
# ValueError: Variable bar/v does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# with tf.variable_scope("bar", reuse=True):
#     v = tf.get_variable("v", [1])
"""
tf.variable_scope嵌套
"""
with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值。
    print(tf.get_variable_scope().reuse)  # 输出为False，即最外层reuse是False
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)  # 输出为True
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)  # 输出为True，新建一个嵌套的上下文管理器但不指定reuse，这是和外层的一致
    print(tf.get_variable_scope().reuse)  # 输出为False，突出reuse=True的上下文后，又回到了False。
"""
使用tf.variable_scope作为变量的命名空间。
"""
p1 = tf.get_variable("p", [1])
print(p1.name)  # 输出为p:0,"p"为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果。
with tf.variable_scope("foo"):
    p2 = tf.get_variable("p", [1])
    print(p2.name)  # 输出foo/p:0,
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        p3 = tf.get_variable("p", [1])
        print(p3.name)  # 输出foo/bar/p:0
    p4 = tf.get_variable("p1", [1])
    print(p4.name)  # 输出foo/p1:0
# 创建一个空的命名空间，并设置reuse=True。
with tf.variable_scope("", reuse=True):
    p5 = tf.get_variable("foo/bar/p", [1])
    print(p5 == p3)  # 输出True
    p6 = tf.get_variable("foo/p1", [1])
    print(p6 == p4)  # 输出True
