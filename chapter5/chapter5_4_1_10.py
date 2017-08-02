import tensorflow as tf
from tensorflow.python.platform import gfile

"""
只需要得到计算图中某个节点的取值，就可以直接计算定义的加法运算的结果
"""
with tf.Session() as sess:
    model_filename = "D:/tmp/model/combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GrapDef Protocol Buffer.
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前的图中。retrun_elements=["add:0"]给出了返回的张量的名称。在保存的时候给出的是计算节点的名称，
    # 所以为“add".在加载的时候给出的是张量的名称，所以是add:0
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
