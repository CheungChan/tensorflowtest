from tensorflow.examples.tutorials.mnist import input_data

"""
载入mnist数据集（包括训练图片、训练答案、测试图片、测试答案），
测试数据集中训练数据、验证数据、测试数据的大小，打印一张图片的数据，打印他的答案
"""
# 载入MNIST数据集，如果指定地址下没有已经下载好的数据，那么TensorFlow会自动下载数据
mnist = input_data.read_data_sets(r"D:\tmp", one_hot=True)
# 打印Training data size: 55000.
print(f"Training data size: {mnist.train.num_examples}")
# 打印Validating data size: 5000.
print(f"Validating data size: {mnist.validation.num_examples}")
# 打印Testing data size: 10000
print(f"Testing data size: {mnist.test.num_examples}")
# 打印Example training data:
print(f"Example training data: {mnist.test.images[0]}")
# 打印Example traning data label:
# [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
print(f"Example training data label{mnist.train.labels[0]}")
"""
使用mnist.train.next_batch(batch_size)方法，可以将数据分为若干个batch
"""
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个训练数据
print(f"X shape: {xs.shape}")
# 输出 X shape: (100, 784)
print(f"Y shape: {ys.shape}")
# 输出 Y shape: (100, 10)
