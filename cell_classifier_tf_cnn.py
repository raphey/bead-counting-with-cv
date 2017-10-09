__author__ = 'raphey'

import numpy as np
import tensorflow as tf
import glob
import cv2
from classifier_util import *
import os


# Suppress sub-optimal speed warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def conv2d(x_tensor, weight, bias, strides=2):
    """
    Convolutional layer with ReLU activation
    """
    x_tensor = tf.nn.conv2d(x_tensor, weight, strides=[1, strides, strides, 1], padding='SAME')
    x_tensor = tf.nn.bias_add(x_tensor, bias)
    return tf.nn.relu(x_tensor)


def conv_net(x_tensor, ws, bs, dropout):
    # Layer 1 - 9x9*1 to 5*5*32
    conv1 = conv2d(x_tensor, ws['wc1'], bs['bc1'])

    # Layer 2 - 5*5*32 to 3*3*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Fully connected layer - 3*3*64 to 512
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - cell likelihood prediction, 0.0 to 1.0.
    out = tf.sigmoid(tf.add(tf.matmul(fc1, weights['out']), biases['out']))
    return out


cell_data = import_data('training_data/set2/cells', extra_dim=True)
non_cell_data = import_data('training_data/set2/non_cells', extra_dim=True)

training, validation, testing = prepare_data(cell_data, non_cell_data, flat=False)

print('Training data loaded and prepared.')

sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 9, 9, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])


# Convolutional filter depths:
depths = [32, 64, 128]

# Weight and bias variables
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[3, 3, 1, depths[0]], stddev=0.06)),
    'wc2': tf.Variable(tf.random_normal(shape=[3, 3, depths[0], depths[1]], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal(shape=[9 * depths[1], depths[2]], stddev=0.1)),
    'out': tf.Variable(tf.random_normal(shape=[depths[2], 1], stddev=0.1))}

biases = {
    'bc1': tf.Variable(tf.zeros([depths[0]])),
    'bc2': tf.Variable(tf.zeros([depths[1]])),
    'bd1': tf.Variable(tf.zeros([depths[2]])),
    'out': tf.Variable(tf.zeros([1]))}


epochs = 30
batch_size = 32
num_batches = len(training[0]) // batch_size
learning_rate = 0.0001
keep_prob = 0.8

# Graph for logits
y = conv_net(x, weights, biases, keep_prob)

print("Graph built")

# Cost
cost = tf.nn.l2_loss(y - y_, name="squared_error_cost")

# Accuracy
correct_pred = tf.equal(tf.round(y), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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

        print('Epoch: {:>4}/{}   Training cost: {:<5.1f}   Pos. & neg. val. acc.: {:.4f},  {:.4f}'.format(
              e + 1, epochs, epoch_loss, pos_valid_acc, neg_valid_acc))

    pos_test_acc = sess.run(accuracy, feed_dict={x: testing[0], y_: testing[1]})
    neg_test_acc = sess.run(accuracy, feed_dict={x: testing[2], y_: testing[3]})

    print('Training complete. Positive & negative testing accuracy: {:.4f},  {:.4f}'.format(
          pos_test_acc, neg_test_acc))

    save_path = saver.save(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")
    print("Model saved in file: %s" % save_path)