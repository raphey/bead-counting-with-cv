__author__ = 'raphey'

import numpy as np
import tensorflow as tf
import glob
import cv2
from classifier_util import *
import os


# Suppress sub-optimal speed warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


cell_data = import_data('training_data/cells')
non_cell_data = import_data('training_data/non_cells')
training, validation, testing = prepare_data(cell_data, non_cell_data, flat=True)

sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Variables for one hidden layer
# w1 = tf.Variable(tf.random_normal(shape=[784, 25], stddev=0.1))
# b1 = tf.Variable(tf.zeros([1]))
# w2 = tf.Variable(tf.random_normal(shape=[25, 1], stddev=0.1))
# b2 = tf.Variable(tf.zeros([1]))
# l2i = tf.sigmoid(tf.matmul(x, w1) + b1)
# y = tf.sigmoid(tf.matmul(l2i, w2) + b2)

# Variables for two hidden layers
w1 = tf.Variable(tf.random_normal(shape=[784, 80], stddev=0.1))
b1 = tf.Variable(tf.zeros([1]))
w2 = tf.Variable(tf.random_normal(shape=[80, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))
w3 = tf.Variable(tf.random_normal(shape=[10, 1], stddev=0.1))
b3 = tf.Variable(tf.zeros([1]))
l2i = tf.sigmoid(tf.matmul(x, w1) + b1)
l3i = tf.sigmoid(tf.matmul(l2i, w2) + b2)
y = tf.sigmoid(tf.matmul(l3i, w3) + b3)


epochs = 25
batch_size = 32
num_batches = len(training[0]) // batch_size
learning_rate = 0.001

# Cost
cost = tf.nn.l2_loss(y - y_, name="squared_error_cost")

# Accuracy
correct_pred = tf.equal(tf.round(y), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# For saving model
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for e in range(epochs):

        epoch_loss = 0.0

        for b in range(num_batches):
            start_index = batch_size * b
            batch_x = training[0][start_index:start_index + batch_size]
            batch_y = training[1][start_index:start_index + batch_size]

            optimizer.run(feed_dict={x: batch_x, y_: batch_y})

            epoch_loss += sess.run(cost, feed_dict={x: batch_x, y_: batch_y})

        pos_valid_acc = sess.run(accuracy, feed_dict={x: validation[0], y_: validation[1]})
        neg_valid_acc = sess.run(accuracy, feed_dict={x: validation[2], y_: validation[3]})

        print('Epoch: {:>4}/{}   Training cost: {:<5.1f}   Pos. & neg. val. acc.: {:.3f},  {:.3f}'.format(
              e + 1, epochs, epoch_loss, pos_valid_acc, neg_valid_acc))

    pos_test_acc = sess.run(accuracy, feed_dict={x: testing[0], y_: testing[1]})
    neg_test_acc = sess.run(accuracy, feed_dict={x: testing[2], y_: testing[3]})

    print('Training complete. Positive & negative testing accuracy: {:.3f},  {:.3f}'.format(
          pos_test_acc, neg_test_acc))

    save_path = saver.save(sess, "classifier_data/tf_model.ckpt")
    print("Model saved in file: %s" % save_path)