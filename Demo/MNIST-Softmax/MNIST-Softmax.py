#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# get MNIST data set
print("get MNIST data......")
mnist = input_data.read_data_sets("MNIST-Data/",one_hot=True)

# create basic Tensor node
print("create basic Tensor node......")
x = tf.placeholder(tf.float32,shape=(None,784))
W = tf.Variable(tf.zeros(shape=(784,10)))
b = tf.Variable(tf.zeros(shape=(10,)))
y_ = tf.placeholder(tf.float32,shape=(None,10))

# create softmax Tensor node
print("create softmax Tensor node......")
y = tf.nn.softmax(tf.matmul(x,W)+b)

# create loss(cross entropy) Tensor node
print("create loss Tensor node......")
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

# create training Operation
print("create training operation......")
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# init variables
print("init global variables......")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# run training operation(stochastic training)
print("start training......")
for _ in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    train.run(feed_dict={x:batch_x,y_:batch_y})
print("end training......")

# create evaluate Tensor node
print("create evaluate Tensor node......")
evaluate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1)),dtype=tf.float32))

# eval evaluate
print("eval softmax model......")
print("Accuracy: %s" % evaluate.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
