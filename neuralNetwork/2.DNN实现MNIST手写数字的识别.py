# -*- coding:utf-8 -*-
_author_ = "PayneLi"
_Time_ = "18-5-13  下午11:10"
_File_ = "2.DNN实现MNIST手写数字的识别.py"
SoftWare = "PyCharm"

"""create your coding"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

INPUT_NODE = 784
HIDE_LAYER_NODE = 500
OUTPUT_NODE = 10

"""1.准备数据"""
data = input_data.read_data_sets(train_dir="./mnist/input_data/", one_hot=True)

"""2.输入层"""
with tf.variable_scope("data_input"):
    x = tf.placeholder(dtype=tf.float32, name="x")
    y = tf.placeholder(dtype=tf.int32, name="y")

"""3.隐藏层"""
with tf.variable_scope("hide_layer"):
    weight_1 = tf.Variable(tf.random_normal(shape=[INPUT_NODE, HIDE_LAYER_NODE]))
    bias_1 = tf.Variable(tf.random_normal(shape=[HIDE_LAYER_NODE]), name="bias_1")
    hide_layer_out = tf.matmul(x, weight_1) + bias_1

"""4.激活函数"""
with tf.variable_scope("relu_out"):
    relu_out = tf.nn.relu(hide_layer_out)

"""5.原始输出层"""
with tf.variable_scope("original_output"):
    weight_2 = tf.Variable(tf.random_normal(shape=[HIDE_LAYER_NODE, OUTPUT_NODE]), name="weight_2")
    bias_2 = tf.Variable(tf.random_normal(shape=[OUTPUT_NODE]))
    logits = tf.matmul(relu_out, weight_2)

"""6.Softmax回归和交叉熵"""
with tf.variable_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

"""7.使用梯度下降训练模型"""
with tf.variable_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)

"""8.计算精度"""
with tf.variable_scope("accuracy"):
    equal_list = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32))

"""9.TensorFlow的可视化以及合并"""
filter_writer = tf.summary.FileWriter(logdir="./log", graph=tf.get_default_graph())

tf.summary.scalar(name="loss", tensor=loss)
tf.summary.scalar(name="accuracy", tensor=accuracy)
tf.summary.histogram(name="weight_1", values=weight_1)
tf.summary.histogram(name="weight_2", values=weight_2)

merge_all = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    image_batch, label_batch = data.train.next_batch(batch_size=50)
    # 50张图片，50张标签(每个标签都是一个one-hot编码)

    for i in range(100):
        sess.run(train_op, feed_dict={x: image_batch, y: label_batch})
        accuracy_result = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch})
        print(accuracy_result)

        summary = sess.run(merge_all, feed_dict={x: image_batch, y: label_batch})
        filter_writer.add_summary(summary, i)
