import tensorflow as tf

with tf.variable_scope("foo"):
    a = tf.get_variable("bar", [1])
    print(a.name)  # foo/bar:0

with tf.variable_scope("bar"):
    b = tf.get_variable("bar", [1])
    print(b.name)  # bar/bar:0

with tf.name_scope("a"):
    a = tf.Variable([1])
    print(a.name)  # a/Variable:0
    a = tf.get_variable("b", [1])
    print(a.name)  # b:0

# with tf.name_scope("b"):
#     tf.get_variable("b", [1])
#     # ValueError: Variable b already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
