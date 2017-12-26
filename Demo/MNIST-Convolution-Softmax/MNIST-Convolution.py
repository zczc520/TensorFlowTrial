import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# define functions to create variable Tensor node
def create_weight_node(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)

def create_bias_node(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# define functions to create convolution & pooling Tensor node
def create_conv2d_node(x,W):
    return tf.nn.conv2d(x,W,strides=(1,1,1,1),padding='SAME')

def create_maxpool_node(x):
    return tf.nn.max_pool(x,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')

# get MNIST data set
print("Get MNIST data......")
mnist = input_data.read_data_sets("MNIST-Data",one_hot=True)

# create input Tensor node
print("Create input node......")
x = tf.placeholder(tf.float32,shape=(None,784))
y_ = tf.placeholder(tf.float32,shape=(None,10))

# First Layer - Convolution Layer
print("Create First Layer......")
W_conv1 = create_weight_node(shape=(5,5,1,32)) # the convolution kernel used by this layer
b_conv1 = create_bias_node(shape=(32,)) # the bias used by this layer
x1 = tf.reshape(x,shape=(-1,28,28,1)) # change x from 2D tensor to 4D tensor
conv1 = tf.nn.relu(create_conv2d_node(x1,W_conv1)+b_conv1)
pool1 = create_maxpool_node(conv1)

# Second Layer - Convolution Layer
print("Create Second Layer......")
W_conv2 = create_weight_node(shape=(5,5,32,64)) # the convolution kernel used by this layer
b_conv2 = create_bias_node(shape=(64,)) # the bias used by this layer
conv2 = tf.nn.relu((create_conv2d_node(pool1,W_conv2)+b_conv2))
pool2 = create_maxpool_node(conv2)

# Third Layer - Connected Layer
print("Create Third Layer......")
x3 = tf.reshape(pool2,shape=(-1,7*7*64))
W_conn3 = create_weight_node(shape=(7*7*64,1024))
b_conn3 = create_bias_node(shape=(1024,))
conn3 = tf.nn.relu(tf.matmul(x3,W_conn3)+b_conn3)

# Fourth Layer - Dropout Layer
print("Create Fourth Layer......")
keep_prob=tf.placeholder(tf.float32) # keep probability
drop4 = tf.nn.dropout(conn3,keep_prob)

# Final Layer - Readout Layer
print("Create Final Layer......")
W_final = create_weight_node(shape=(1024,10))
b_final = create_bias_node(shape=(10,))
final = tf.nn.relu(tf.matmul(drop4,W_final)+b_final)

# create optimizer
print("Create optimizer......")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final,labels=y_))
optimizer = tf.train.AdamOptimizer(1e-4)
min_operation = optimizer.minimize(loss)

# create accuracy evaluater
print("Create accuracy evaluater......")
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final,1),tf.argmax(y_,1)),tf.float32))

# train and evaluate
filename = "MNIST-Convolution.log"
logfile = open(filename,'w')
with tf.Session() as sess:
    print("Start trainning......")
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batchx,batchy = mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy = sess.run(accuracy,feed_dict={x:batchx,y_:batchy,keep_prob:1})
            logfile.write('Step %d: accuracy %g\n' % (i,train_accuracy))
        sess.run(min_operation,feed_dict={x:batchx,y_:batchy,keep_prob:0.5})
    print("Start test......")
    test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
    logfile.write('Final accuracy: %g' % test_accuracy)

# close objects
logfile.close()

