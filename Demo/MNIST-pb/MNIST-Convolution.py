import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import os

# global configure
DATA_DIR = "MNIST-Data"
MODEL_DIR = "Models"
PB_FILENAME = "CNN.pb"
LOG_FILENAME = "MNIST-Convolution.log"
BATCH_SIZE = 50
TRAIN_TIMES = 100
KEEP_PROB = 0.5
# variable
W_CONV1_NAME = "w_conv1"
B_CONV1_NAME = "b_conv1"
W_CONV2_NAME = "w_conv2"
B_CONV2_NAME = "b_conv2"
W_CONN3_NAME = "w_conn3"
B_CONN3_NAME = "b_conn3"
W_FINAL5_NAME = "w_final5"
B_FINAL5_NAME = "b_final5"
# placeholder
INPUT_NAME = "x"
LABEL_NAME = "y_"
KEEP_PROB_NAME = "keep_prob"

# define functions to create variable Tensor node
def create_weight_node(shape,name):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def create_bias_node(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

# define functions to create convolution & pooling Tensor node
def create_conv2d_node(x,W,):
    return tf.nn.conv2d(x,W,strides=(1,1,1,1),padding='SAME')

def create_maxpool_node(x,name):
    return tf.nn.max_pool(x,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME',name=name)

# get MNIST data set
print("Get MNIST data......")
mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)

# create input Tensor node
print("Create input node......")
x = tf.placeholder(tf.float32,shape=(None,784),name=INPUT_NAME)
y_ = tf.placeholder(tf.float32,shape=(None,10),name=LABEL_NAME)

# First Layer - Convolution Layer
print("Create First Layer......")
W_conv1 = create_weight_node(shape=(5,5,1,32),name=W_CONV1_NAME) # the convolution kernel used by this layer
b_conv1 = create_bias_node(shape=(32,),name=B_CONV1_NAME) # the bias used by this layer
x1 = tf.reshape(x,shape=(-1,28,28,1)) # change x from 2D tensor to 4D tensor
conv1 = tf.nn.relu(create_conv2d_node(x1,W_conv1)+b_conv1)
pool1 = create_maxpool_node(conv1,name="testpool1")

# Second Layer - Convolution Layer
print("Create Second Layer......")
W_conv2 = create_weight_node(shape=(5,5,32,64),name=W_CONV2_NAME) # the convolution kernel used by this layer
b_conv2 = create_bias_node(shape=(64,),name=B_CONV2_NAME) # the bias used by this layer
conv2 = tf.nn.relu((create_conv2d_node(pool1,W_conv2)+b_conv2))
pool2 = create_maxpool_node(conv2,name="testpool2")

# Third Layer - Connected Layer
print("Create Third Layer......")
x3 = tf.reshape(pool2,shape=(-1,7*7*64))
W_conn3 = create_weight_node(shape=(7*7*64,1024),name=W_CONN3_NAME)
b_conn3 = create_bias_node(shape=(1024,),name=B_CONN3_NAME)
conn3 = tf.nn.relu(tf.matmul(x3,W_conn3)+b_conn3,name="testrelu3")

# Fourth Layer - Dropout Layer
print("Create Fourth Layer......")
keep_prob=tf.placeholder(tf.float32,name=KEEP_PROB_NAME) # keep probability
drop4 = tf.nn.dropout(conn3,keep_prob,name="drop")

# Final Layer - Readout Layer
print("Create Final Layer......")
W_final = create_weight_node(shape=(1024,10),name=W_FINAL5_NAME)
b_final = create_bias_node(shape=(10,),name=B_FINAL5_NAME)
final = tf.nn.relu(tf.matmul(drop4,W_final)+b_final,name="final")

# create optimizer
print("Create optimizer......")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final,labels=y_))
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

# create accuracy evaluater
print("Create accuracy evaluater......")
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final,1),tf.argmax(y_,1)),tf.float32),name="accuracy")

# train, save and evaluate
# train
logfile = open(LOG_FILENAME,'w')
with tf.Session() as sess:
    print("Start trainning......")
    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_TIMES):
        batchx,batchy = mnist.train.next_batch(BATCH_SIZE)
        if i%100==0:
            train_accuracy = sess.run(accuracy,feed_dict={x:batchx,y_:batchy,keep_prob:1})
            logfile.write('Step %d: accuracy %g\n' % (i,train_accuracy))
        sess.run(train_step,feed_dict={x:batchx,y_:batchy,keep_prob:KEEP_PROB})

    # save
    print("Save trained model as pb file......")
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['accuracy'])
    if not tf.gfile.Exists(MODEL_DIR):
    	tf.gfile.MakeDirs(MODEL_DIR)
    pb_file_path = os.path.join(MODEL_DIR,PB_FILENAME)
    with tf.gfile.FastGFile(pb_file_path,mode="wb") as fd:
    	fd.write(constant_graph.SerializeToString())

    # evaluate
    print("Start test......")
    test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
    logfile.write('Final accuracy: %g' % test_accuracy)

# close objects
logfile.close()

