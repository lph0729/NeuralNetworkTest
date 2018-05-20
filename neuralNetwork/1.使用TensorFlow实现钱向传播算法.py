# -*- coding:utf-8 -*-
_author_ = "PayneLi"
_Time_ = "18-5-13  下午6:59"
_File_ = "1.使用TensorFlow实现钱向传播算法.py"
SoftWare = "PyCharm"

"""create your coding"""
import tensorflow as tf

# 定义两个权重
w1 = tf.Variable(tf.random_normal(shape=[2, 3]))
w2 = tf.Variable(tf.random_normal(shape=[3, 1]))

# 定义输入ｘ
x = tf.constant([1, 1], shape=[1, 2], dtype=tf.float32)

a = tf.nn.relu(tf.matmul(x, w1))
y = tf.nn.relu(tf.matmul(a, w2))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run(y)
    print(result)
