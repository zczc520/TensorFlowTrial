from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
DATA_DIR = "IRIS_Data"
MODEL_DIR = "Models" # save models and summary file
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_trainging.csv"
IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # Download data set
    train_filename = os.path.join(DATA_DIR,IRIS_TRAINING)
    test_filename = os.path.join(DATA_DIR,IRIS_TEST)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(train_filename):
        urllib.urlretrieve(IRIS_TRAINING_URL,train_filename)
    if not os.path.exists(test_filename):
        urllib.urlretrieve(IRIS_TEST_URL,test_filename)

    # Load data set
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=train_filename,
            target_dtype=np.int,
            features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=test_filename,
            target_dtype=np.int,
            features_dtype=np.float32)

    # Build model
    feature_columns = [tf.feature_column.numeric_column("x",shape=[4])]
    DNN = tf.estimator.DNNClassifier(feature_columns=feature_columns,
            hidden_units=[10,20,10],
            n_classes=3,
            model_dir=MODEL_DIR)

    # Prepare input data set for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True)

    # Train
    DNN.train(input_fn=train_input_fn,steps=2000)

    # Prepare input data set for testing
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":np.array(test_set.data)},
            y=np.array(test_set.target),
            num_epochs=1,
            shuffle=False)

    # Test
    test_result = DNN.evaluate(input_fn=test_input_fn)
    for x,y in test_result.items():
        print("{}: {}".format(x,y))

    # Inference
    new_samples = np.array(
            [[6.4,3.2,4.5,1.5],
             [5.8,3.1,5.0,1.7]])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":new_samples},
            num_epochs=1,
            shuffle=False)
    predictions = list(DNN.predict(input_fn=predict_input_fn))
    for p in predictions:
        for x,y in p.items():
            print("{}: {}".format(x,y)),
        print("")

if __name__ == "__main__":
    main()
