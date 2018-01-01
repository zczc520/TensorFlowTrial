import csv
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import collections

DATA_DIR = "IRIS_Data"
TRAIN_FILE = os.path.join(DATA_DIR,"iris_training.csv")
TEST_FILE = os.path.join(DATA_DIR,"iris_test.csv")
BATCH_SIZE = 100

Dataset = collections.namedtuple('Dataset',['data','target'])

# change class 0 into class -1, remove class 2
def change_data(dataset):
    newdata = np.array(dataset.data)
    newtarget = np.array(dataset.target)
    index = 0
    for v in dataset.target:
        if v == 0:
            newtarget[index] = -1
            index += 1
        elif v == 1:
            index += 1
        else:
            newtarget = np.delete(newtarget,index)
            newdata = np.delete(newdata,index,axis=0)
    return Dataset(data=newdata,target=newtarget)

# load Iris data set for binary classify (only keep class 0 and class 1)
def load_data():
    trainset = base.load_csv_with_header(TRAIN_FILE,target_dtype=np.float32,features_dtype=np.float32)
    testset = base.load_csv_with_header(TEST_FILE,target_dtype=np.float32,features_dtype=np.float32)
    trainset = change_data(trainset)
    testset = change_data(testset)
    return (trainset,testset)

# get batch data
def get_batch_data(dataset):
    indexes = np.random.choice(dataset.data.shape[0],BATCH_SIZE)
    return (dataset.data[indexes],dataset.target[indexes])

def main(argv):
    # load Iris data set with only class -1(0 originally) and +1
    trainset,testset = load_data()

    with tf.Graph().as_default():
        # build Linear SVM Model
        x = tf.placeholder(shape=[None,4],dtype=tf.float32)
        y_ = tf.placeholder(shape=[None,1],dtype=tf.float32)
        A = tf.Variable(tf.random_normal(shape=[4,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))
        model_out = tf.matmul(x,A)+b

        # build loss function
        alpha = tf.constant(0.1)
        square_term = tf.reduce_sum(tf.square(A))*alpha
        restrict_term = tf.reduce_mean(tf.maximum(0.,1-y_*model_out))
        loss = restrict_term + square_term

        # build train step
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # build eval node
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(model_out),y_),tf.float32))

        with tf.Session() as sess:
            # init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # train
            for i in range(2000):
                batchx,batchy = get_batch_data(trainset)
                sess.run(train_step,feed_dict={x:batchx,y_:np.transpose([batchy])})
                # eval
                if (i+1)%100 == 0:
                    loss_value = sess.run(loss,feed_dict={x:testset.data,y_:np.transpose([testset.target])})
                    accur_value = sess.run(accuracy,feed_dict={x:testset.data,y_:np.transpose([testset.target])})
                    print("accuracy:{} loss:{}".format(accur_value,loss_value))

if __name__ == "__main__":
    tf.app.run(main=main)
