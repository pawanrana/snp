from libs.utils import load_pandas
import StringIO
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from snp import tf_confusion_metrics,training_predictors_tf,training_classes_tf,test_predictors_tf,test_classes_tf

#changeing structure of NN into accepting parameters, instead of hard coding
#implementing random initilize
#implementing population production

def train_NN(training_predictors_tf,training_classes_tf,test_predictors_tf,test_classes_tf):
  y = training_predictors_tf
  c = training_classes_tf
  y_ = test_predictors_tf
  c_ = test_classes_tf
  input_len = len(y.columns)

  sess = tf.Session()

  num_predictors = len(training_predictors_tf.columns)
  num_classes = len(training_classes_tf.columns)

  feature_data = tf.placeholder("float", [None, num_predictors])
  actual_classes = tf.placeholder("float", [None, 2])

  weights1 = tf.Variable(tf.truncated_normal([input_len, 50], stddev=0.0001))
  biases1 = tf.Variable(tf.ones([50]))

  weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
  biases2 = tf.Variable(tf.ones([25]))
                       
  weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
  biases3 = tf.Variable(tf.ones([2]))

  hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
  hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
  model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

  cost = -tf.reduce_sum(actual_classes*tf.log(model))

  train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

  init = tf.initialize_all_variables()
  sess.run(init)

  correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  for i in range(1, 30001):
    sess.run(
      train_op1, 
      feed_dict={
        feature_data: training_predictors_tf.values, 
        actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
      }
    )
    if i%5000 == 0:
      print i, sess.run(
        accuracy,
        feed_dict={
          feature_data: training_predictors_tf.values, 
          actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        }
      )

  feed_dict= {
    feature_data: test_predictors_tf.values,
    actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)
  }

  tf_confusion_metrics(model, actual_classes, sess, feed_dict)

  return accuracy
