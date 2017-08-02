import tensorflow as tf
from tensorflow.python.framework import graph_util

"""
使用TensorFlow提供的convert_variables_to_constants函数，通过这个函数可以将计算图中的变量和值通过常量的方式保存，这样整个
TensorFlow计算图可以统一存放在一个文件中。
"""

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 到处当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，同时将途中不必要的节点去掉。一些系统运算也会转换为节点（比如变量初始化操作）。如果只关心程序中
    # 定义的某些运算时，和这些计算无关的节点就没必要导出并保存了。在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。add
    # 节点是上面定义的两个变量相加的操作。注意这里给出的计算节点的名称，所以没有后面的:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    # 将导出的模型存入文件。
    with tf.gfile.GFile(r"D:/tmp/model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
