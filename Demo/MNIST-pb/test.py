import xml.dom.minidom
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data

dom=xml.dom.minidom.parse("conf.xml")
root=dom.documentElement
DATA_DIR=root.getElementsByTagName('DATA_DIR')[0].firstChild.data
MODEL_DIR=root.getElementsByTagName('MODEL_DIR')[0].firstChild.data
PB_FILENAME=root.getElementsByTagName('PB_FILENAME')[0].firstChild.data
LOG_FILENAME=root.getElementsByTagName('LOG_FILENAME')[0].firstChild.data
BATCH_SIZE=root.getElementsByTagName('BATCH_SIZE')[0].firstChild.data
TRAIN_TIMES=root.getElementsByTagName('TRAIN_TIMES')[0].firstChild.data
KEEP_PROB=root.getElementsByTagName('KEEP_PROB')[0].firstChild.data
W_CONV1_NAME=root.getElementsByTagName('W_CONV1_NAME')[0].firstChild.data
B_CONV1_NAME=root.getElementsByTagName('B_CONV1_NAME')[0].firstChild.data
W_CONV2_NAME=root.getElementsByTagName('W_CONV2_NAME')[0].firstChild.data
B_CONV2_NAME=root.getElementsByTagName('B_CONV2_NAME')[0].firstChild.data
W_CONN3_NAME=root.getElementsByTagName('W_CONN3_NAME')[0].firstChild.data
B_CONN3_NAME=root.getElementsByTagName('B_CONN3_NAME')[0].firstChild.data
W_FINAL5_NAME=root.getElementsByTagName('W_FINAL5_NAME')[0].firstChild.data
B_FINAL5_NAME=root.getElementsByTagName('B_FINAL5_NAME')[0].firstChild.data
INPUT_NAME=root.getElementsByTagName('INPUT_NAME')[0].firstChild.data
LABEL_NAME=root.getElementsByTagName('LABEL_NAME')[0].firstChild.data
KEEP_PROB_NAME=root.getElementsByTagName('KEEP_PROB_NAME')[0].firstChild.data

model_path = os.path.join(MODEL_DIR,PB_FILENAME)

# get mnist data
mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)

# load model from pb file
with tf.Session() as sess:
	output_graph_def = tf.GraphDef()
	with open(model_path,'rb') as fd:
		output_graph_def.ParseFromString(fd.read())
		_ = tf.import_graph_def(output_graph_def,name="")

	# print variables value
	log_file = open("test.log",'w')
	w1 = sess.graph.get_tensor_by_name(W_CONV1_NAME+':0')
	log_file.write("w1:{}\n".format(sess.run(w1)))
	b1 = sess.graph.get_tensor_by_name(B_CONV1_NAME+':0')
	log_file.write("b1:{}\n".format(sess.run(b1)))
	w2 = sess.graph.get_tensor_by_name(W_CONV2_NAME+':0')
	log_file.write("w2:{}\n".format(sess.run(w2)))
	b2 = sess.graph.get_tensor_by_name(B_CONV2_NAME+':0')
	log_file.write("b2:{}\n".format(sess.run(b2)))
	w3 = sess.graph.get_tensor_by_name(W_CONN3_NAME+':0')
	log_file.write("w3:{}\n".format(sess.run(w3)))
	b3 = sess.graph.get_tensor_by_name(B_CONN3_NAME+':0')
	log_file.write("b3:{}\n".format(sess.run(b3)))
	w5 = sess.graph.get_tensor_by_name(W_FINAL5_NAME+':0')
	log_file.write("w5:{}\n".format(sess.run(w5)))
	b5 = sess.graph.get_tensor_by_name(B_FINAL5_NAME+':0')
	log_file.write("b5:{}\n".format(sess.run(b5)))

	# print evaluate result
	x = sess.graph.get_tensor_by_name(INPUT_NAME+":0")
	y_ = sess.graph.get_tensor_by_name(LABEL_NAME+":0")
	keep_prob = sess.graph.get_tensor_by_name(KEEP_PROB_NAME+":0")
	accuracy = sess.graph.get_tensor_by_name("accuracy:0")
	result = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
	log_file.write("Final accuracy: {}\n".format(result))

	log_file.close()

