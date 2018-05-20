# -*- coding:utf-8 -*-
_author_ = "PayneLi"
_Time_ = "18-5-16  下午10:35"
_File_ = "3.CNN实现MNIST手写数字识别.py"
SoftWare = "PyCharm"

"""create your coding"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

"""1.准备数据"""
data = input_data.read_data_sets(train_dir="./mnist/input_data/", one_hot=True)

"""2."输入层"""
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.int32)

"""3.卷积层及激活函数：conv_1、relu_1"""
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
filter_1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 32]))
conv_1 = tf.nn.conv2d(x_image, filter_1, strides=[1, 1, 1, 1], padding="SAME")
bias_1 = tf.Variable(tf.random_normal(shape=[32]))
relu_1 = tf.nn.relu(conv_1 + bias_1)
print("---------relu_1-------------------", relu_1)

"""4.池化层1"""
pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

print("-------------pool_1----------------------", pool_1)

"""5.卷积层以及激活函数:conv_2、relu_2"""
filter_2 = tf.Variable(tf.random_normal(shape=[5, 5, 32, 64]))
conv_2 = tf.nn.conv2d(pool_1, filter_2, strides=[1, 1, 1, 1], padding="SAME")
bias_2 = tf.Variable(tf.random_normal(shape=[64]))
relu_2 = tf.nn.relu(conv_2 + bias_2)

print("---------relu_2-------------------", relu_2)

"""6.池化层2"""
pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

print("-------------pool_2----------------------", pool_2)

"""7.全连接层以及激活函数"""
fc_input = tf.reshape(pool_2, shape=[-1, 7 * 7 * 64])
fc_weight_1 = tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 1024]))
fc_bias_1 = tf.Variable(tf.random_normal(shape=[1024]))
logits_1 = tf.matmul(fc_input, fc_weight_1) + fc_bias_1
fc_relu = tf.nn.relu(logits_1)

"""8.原始输出层"""
fc_weight_2 = tf.Variable(tf.random_normal(shape=[1024, 10]))
fc_bias_2 = tf.Variable(tf.random_normal(shape=[10]))
logits_2 = tf.matmul(fc_relu, fc_weight_2) + fc_bias_2

"""9.Softmax回归与交叉熵"""
sotfmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_2)
loss = tf.reduce_mean(sotfmax_loss)

"""10.使用梯度下降训练模型、并且计算准确率"""
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
equal_list = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_2, axis=1))
accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        image_batch, label_batch = data.train.next_batch(batch_size=50)
        sess.run(train_op, feed_dict={x: image_batch, y: label_batch})
        accurity_ret = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch})
        print("----------------accurity_ret--------------------", accurity_ret)
