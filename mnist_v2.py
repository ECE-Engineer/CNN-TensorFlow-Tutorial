from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import platform

# set constants
TOTAL_EPOCHS = 1000
label_count = 10
data_width = 28
data_height = 28
batch_size = 64
total_train_data = None
total_test_data = None
log_dir = os.getcwd()
generic_slash = None
if platform.system() == 'Windows':
  generic_slash = '\\'
else:
  generic_slash = '/'

def encodeLabels(labels_decoded):
    encoded_labels = np.zeros(shape=(len(labels_decoded), label_count), dtype=np.int8)
    for x in range(0, len(labels_decoded)):
        some_label = labels_decoded[x]

        if 0 == some_label:
            encoded_labels[x][0] = 1
        elif 1 == some_label:
            encoded_labels[x][1] = 1
        elif 2 == some_label:
            encoded_labels[x][2] = 1
        elif 3 == some_label:
            encoded_labels[x][3] = 1
        elif 4 == some_label:
            encoded_labels[x][4] = 1
        elif 5 == some_label:
            encoded_labels[x][5] = 1
        elif 6 == some_label:
            encoded_labels[x][6] = 1
        elif 7 == some_label:
            encoded_labels[x][7] = 1
        elif 8 == some_label:
            encoded_labels[x][8] = 1
        elif 9 == some_label:
            encoded_labels[x][9] = 1
    return encoded_labels

def weight_variable(shape):
  # uses default std. deviation
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  # uses default bias
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def shallow_cnn(x):
  # Reshape to use within a convolutional neural net
  x_image = tf.reshape(x, [-1, data_width, data_height, 1])
  
  # Define Regularizer
  regularizer = tf.keras.constraints.MaxNorm(max_value=2)

  # Convolutional Layer #1
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, regularizer.__call__(w=W_conv1), 
      strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # Pooling Layer #2
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Convolutional Layer #2
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, regularizer.__call__(w=W_conv2), 
      strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
  
  # Batch Norm
  batch_norm = tf.layers.batch_normalization(h_conv2,momentum=0.1,epsilon=1e-5)

  # Pooling Layer #2
  h_pool2 = tf.nn.max_pool(batch_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Dense Layer
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  
  # Flatten
  h_pool1_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
  
  # Dropout
  keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Logits Layer
  W_fc2 = weight_variable([1024, label_count])
  b_fc2 = bias_variable([label_count])
  
  # Max Norm
  regularizer = tf.keras.constraints.MaxNorm(max_value=0.5)

  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, regularizer.__call__(w=W_fc2)) + b_fc2, name = "y_conv")

  return y_conv, keep_prob

# Create a method to run the model, save it, & get statistics
def run_model():
  # Load training and eval data
  print("Data Loading")
  mnist = tf.keras.datasets.mnist
  (train_x, train_y),(test_x, test_y) = mnist.load_data()
  train_x, test_x = train_x / 255.0, test_x / 255.0

  total_train_data = len(train_y)
  total_test_data = len(test_y)

  print("Encoding Labels")
  # One-Hot encode the labels
  train_y = encodeLabels(train_y)
  test_y = encodeLabels(test_y)

  print("Creating Datasets")
  # Create the DATASETs
  train_x_dataset = tf.data.Dataset.from_tensor_slices(train_x)
  train_y_dataset = tf.data.Dataset.from_tensor_slices(train_y)
  test_x_dataset = tf.data.Dataset.from_tensor_slices(test_x)
  test_y_dataset = tf.data.Dataset.from_tensor_slices(test_y)

  print("Zipping The Data Together")
  # Zip the data and batch it and (shuffle)
  train_data = tf.data.Dataset.zip((train_x_dataset, train_y_dataset)).shuffle(buffer_size=total_train_data).repeat().batch(batch_size).prefetch(buffer_size=5)
  test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(batch_size).prefetch(buffer_size=1)

  print("Creating Iterators")
  # Create Iterators
  train_iterator = train_data.make_initializable_iterator()
  test_iterator = test_data.make_initializable_iterator()

  # Create iterator operation
  train_next_element = train_iterator.get_next()
  test_next_element = test_iterator.get_next()

  print("Defining Model Placeholders")
  # Create the model
  x = tf.placeholder(tf.float32, [None, data_width, data_height], name = "x")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int8, [None, label_count], name = "x")

  # Build the graph for the deep net
  y_conv, keep_prob = shallow_cnn(x)

  # Create loss op
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Create train op
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  CNN_prediction_label = tf.argmax(y_conv, 1)
  actual_label = tf.argmax(y_, 1)
  correct_prediction = tf.equal(CNN_prediction_label, actual_label)

  # Create accuracy op
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Initialize and Run
  with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + generic_slash + tensorflow + generic_slash + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + generic_slash + tensorflow + generic_slash + 'test')
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    saver = tf.train.Saver()
    print("----------------------|----\---|-----|----/----\---|-----|---|\----|---------------------")
    print("----------------------|    |---|     ----|     -------|------|-\---|---------------------")
    print("----------------------|   |----|-----|---|   ---------|------|--\--|---------------------")
    print("----------------------|    |---|     ----|     |------|------|---\-|---------------------")
    print("----------------------|----/---|-----|----\----/---|-----|---|----\|---------------------")
    for i in range(TOTAL_EPOCHS + 1):
      if i % 100 == 0:
        validation_batch = sess.run(test_next_element)
        summary, acc = sess.run([merged, accuracy], feed_dict={
            x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})
        print('step ' + str(i) + ', test accuracy' + str(acc))
        # Save the model
        saver.save(sess, log_dir + generic_slash + tensorflow + generic_slash + "mnist_model.ckpt")
        # Save the summaries
        test_writer.add_summary(summary, i)
        test_writer.flush()
      print("epoch " + str(i))
      batch = sess.run(train_next_element)
      summary, _ = sess.run([merged, train_step], feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 0.5})
      train_writer.add_summary(summary, i)
      train_writer.flush()
    # Evaluate over the entire test dataset
    # Re-initialize
    test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
    test_iterator = test_data.make_initializable_iterator()
    test_next_element = test_iterator.get_next()
    sess.run(test_iterator.initializer)
    
    # Run for final accuracy
    validation_batch = sess.run(test_next_element)
    print('Final Accuracy ' + str(accuracy.eval(feed_dict={
        x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})))
    print("FINISHED")
    
    # Re-initialize
    test_data = tf.data.Dataset.zip((test_x_dataset, test_y_dataset)).shuffle(buffer_size=total_test_data).repeat().batch(total_test_data).prefetch(buffer_size=1)
    test_iterator = test_data.make_initializable_iterator()
    test_next_element = test_iterator.get_next()
    sess.run(test_iterator.initializer)
    
    print("Creating Confusion Matrix")
    predict, correct = sess.run([CNN_prediction_label, actual_label], feed_dict={
        x: validation_batch[0], y_: validation_batch[1], keep_prob: 1.0})
    skplt.metrics.plot_confusion_matrix(correct, predict, normalize=True)
    plt.savefig(log_dir + generic_slash + tensorflow + generic_slash + "plot.png")
    
# Create a loader for the graph
def graph_loader():
  with tf.Session() as sess:
    #load the graph
	restore_saver = tf.train.import_meta_graph(log_dir + generic_slash + 'tensorflow' 
	    + generic_slash + "mnist_model.ckpt")
	#reload all the params to the graph
	restore_saver.restore(sess, tf.train.latest_checkpoint(log_dir))
	global model
	model = tf.get_default_graph()
	
	#store the variables
	global x
	x = graph.get_tensor_by_name("x:0")
	global y_
	y_ = graph.get_tensor_by_name("y_:0")
	global y_conv
	y_conv = graph.get_tensor_by_name("y_conv:0")
	global keep_prob
	keep_prob = graph.get_tensor_by_name("keep_prob:0")